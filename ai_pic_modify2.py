"""
ai_pic_modify2.py
=================
批量 AI 图片处理工具（混元生图 3.0 版）：
- 使用 LangChain ChatOpenAI（对接腾讯混元 OpenAI 兼容接口）生成图像编辑指令
- 使用腾讯云 aiart SubmitTextToImageJob（混元生图 3.0）接口执行图生图
  - 异步接口：提交任务获取 JobId → 轮询 QueryTextToImageJob 直至完成 → 下载图片
- 通过 Images 参数传入参考图（Base64），实现图生图
- 保留衣服细节，去除水印，改变模特姿势
- 结果保存到带时间日期的子目录

使用方式：
    uv run python ai_pic_modify2.py

.env 文件（项目根目录）：
    TENCENT_SECRET_ID   - 腾讯云 SecretId
    TENCENT_SECRET_KEY  - 腾讯云 SecretKey
"""

from __future__ import annotations

import base64
import logging
import os
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from tencentcloud.aiart.v20221229 import aiart_client, models
from tencentcloud.common import credential
from tencentcloud.common.exception.tencent_cloud_sdk_exception import TencentCloudSDKException

# ---------------------------------------------------------------------------
# 初始化
# ---------------------------------------------------------------------------

load_dotenv()  # 读取项目根目录的 .env 文件

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}

# 腾讯混元 OpenAI 兼容接口地址
HUNYUAN_BASE_URL = "https://api.hunyuan.cloud.tencent.com/v1"
# 混元支持图文的视觉模型
HUNYUAN_MODEL = "hunyuan-turbos-latest"

# 任务轮询配置
POLL_INTERVAL_SECONDS = 5      # 每次轮询间隔（秒）
POLL_MAX_WAIT_SECONDS = 300    # 最大等待时长（秒，5分钟）

# 任务状态码（QueryTextToImageJob 返回）
JOB_STATUS_WAITING = "1"     # 等待中
JOB_STATUS_RUNNING = "2"     # 运行中
JOB_STATUS_FAILED  = "4"     # 处理失败
JOB_STATUS_DONE    = "5"     # 处理完成

# ---------------------------------------------------------------------------
# LangChain Chain：Prompt → 混元 LLM → 解析
# ---------------------------------------------------------------------------

_SYSTEM_MSG = """\
你是专业的时尚电商图片处理助手。
根据用户提供的图片文件名，生成一段用于腾讯云混元生图 3.0（图生图）接口的图像编辑 Prompt。
严格要求：
1. 去除图片中所有水印、文字覆盖和 logo，还原干净背景
2. 改变模特站姿，使其更自然优雅（侧身展示、手部姿势自然变化）
3. 严格保持模特所穿衣服的颜色、款式、面料、图案和版型完全不变
4. 整体画质清晰，细节丰富，专业商品展示风格
只输出 Prompt 本身，中文，不超过 256 字，不要有任何额外解释。"""

_HUMAN_MSG = "图片文件名：{filename}"


def build_prompt_chain(api_key: str) -> Runnable:
    """
    构建 LangChain Prompt 生成链。

    使用 ChatOpenAI 接入腾讯混元 OpenAI 兼容接口，
    通过 | 管道运算符组成：prompt_template | llm | parser 的标准 LCEL 链。

    Args:
        api_key: 腾讯混元 API Key

    Returns:
        可调用的 LangChain Chain，invoke({"filename": ...}) 返回 Prompt 字符串
    """
    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", _SYSTEM_MSG),
            ("human", _HUMAN_MSG),
        ]
    )

    llm = ChatOpenAI(
        model=HUNYUAN_MODEL,
        api_key=api_key,
        base_url=HUNYUAN_BASE_URL,
        temperature=0.7,
        max_tokens=300,
    )

    # LCEL 管道：prompt | llm | 解析为字符串
    chain = prompt_template | llm | StrOutputParser()
    return chain


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def image_to_base64(image_path: Path) -> str:
    """读取图片文件并转为 Base64 字符串。"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def download_image(url: str, output_path: Path) -> bool:
    """
    从 URL 下载图片并保存到本地文件。

    Returns:
        True = 成功，False = 失败
    """
    try:
        urllib.request.urlretrieve(url, str(output_path))
        return True
    except Exception as exc:
        logger.error("下载图片失败 [%s]: %s", url, exc)
        return False


# ---------------------------------------------------------------------------
# 核心处理器
# ---------------------------------------------------------------------------

class HunyuanImageProcessor:
    """
    腾讯混元生图 3.0 图片处理器。

    - 使用 LangChain ChatOpenAI（混元兼容接口）动态生成图像编辑 Prompt
    - 使用腾讯云 aiart SubmitTextToImageJob 接口提交图生图任务
    - 轮询 QueryTextToImageJob 等待任务完成，下载结果图片
    """

    def __init__(
        self,
        secret_id: str,
        secret_key: str,
        resolution: str = "1024:1024",
    ) -> None:
        """
        Args:
            secret_id:    腾讯云 SecretId（用于 aiart SDK 鉴权）
            secret_key:   腾讯云 SecretKey（同时作为混元 OpenAI 兼容接口的 API Key）
            resolution:   输出图片分辨率，格式 "宽:高"，默认 "1024:1024"
                          支持：768:768、768:1024、1024:768、1024:1024、
                                720:1280、1280:720 等
        """
        # aiart 图像生成客户端（用 SecretId + SecretKey 鉴权）
        cred = credential.Credential(secret_id, secret_key)
        self.aiart_client = aiart_client.AiartClient(cred, "ap-guangzhou")

        api_key = os.environ.get("API_KEY", "").strip()
        # LangChain Chain（用 SecretKey 作为混元 OpenAI 兼容接口的 API Key）
        self.prompt_chain = build_prompt_chain(api_key=api_key)
        self.resolution = resolution

    def _generate_prompt(self, filename: str) -> str:
        """
        调用 LangChain Chain 动态生成当前图片的编辑 Prompt。

        链路：ChatPromptTemplate | ChatOpenAI(混元) | StrOutputParser
        """
        # try:
        #     prompt = self.prompt_chain.invoke({"filename": filename})
        #     logger.info("LangChain 生成 Prompt [%s]: %s", filename, prompt)
        #     return prompt.strip()
        # except Exception as exc:
        #     logger.warning("LangChain 生成 Prompt 失败，使用默认 Prompt。原因：%s", exc)
        #     return (
        #         "高清时尚模特图，去除图片中所有水印、文字覆盖和logo，"
        #         "修改模特的站姿使其更自然优雅，"
        #         "严格保持模特所穿衣服的颜色、款式、面料、图案和版型完全不变，"
        #         "整体画质清晰，细节丰富，专业商品展示风格"
        #     )
        return (
            # "高清时尚模特图，去除图片中所有水印、文字覆盖和logo，"
            # "修改模特的姿态使其更自然和性感，"
            # "严格保持模特所穿衣服的颜色、款式、面料、图案和版型完全不变，"
            # "整体画质清晰，细节丰富，专业商品展示风格, "
            # "如果图像有人脸，将人脸改成亚洲美女风格"
            "请你去除图片中的水印，人物改成亚洲面孔的美女，姿势变换成优雅自然的姿势，姿势改动要大,生成非常清晰的图片，不要改变衣服的任何细节！"
            # "对于NSFW的部位，请自然遮挡防止被限制输出"
        )

    def _submit_image_to_image_job(
        self,
        image_path: Path,
        prompt: str,
    ) -> str | None:
        """
        提交混元生图 3.0 图生图任务（SubmitTextToImageJob）。

        通过 Images 参数传入参考图 Base64，结合 Prompt 实现图生图。

        Returns:
            JobId 字符串（成功），或 None（失败）
        """
        try:
            req = models.SubmitTextToImageJobRequest()
            req.Prompt = prompt
            req.Images = [image_to_base64(image_path)]   # 参考图 Base64 列表
            req.Resolution = self.resolution
            req.LogoAdd = 0     # 不添加 AI 水印标识
            req.Revise = 1      # 开启 prompt 改写，提升生图质量

            response = self.aiart_client.SubmitTextToImageJob(req)
            job_id = response.JobId
            logger.info("  → 任务已提交，JobId: %s", job_id)
            return job_id

        except TencentCloudSDKException as exc:
            logger.error("提交任务失败 [%s]: %s", image_path.name, exc)
            return None
        except Exception as exc:
            logger.error("提交任务异常 [%s]: %s", image_path.name, exc)
            return None

    def _poll_job_result(self, job_id: str) -> list[str] | None:
        """
        轮询 QueryTextToImageJob 直到任务完成，返回结果图片 URL 列表。

        状态码：1=等待中、2=运行中、4=失败、5=完成

        Returns:
            图片 URL 列表（成功），或 None（失败/超时）
        """
        waited = 0
        while waited < POLL_MAX_WAIT_SECONDS:
            time.sleep(POLL_INTERVAL_SECONDS)
            waited += POLL_INTERVAL_SECONDS

            try:
                req = models.QueryTextToImageJobRequest()
                req.JobId = job_id
                response = self.aiart_client.QueryTextToImageJob(req)

                status = response.JobStatusCode
                logger.info(
                    "  → 轮询任务 [%s] 状态: %s（%s），已等待 %ds",
                    job_id, status, response.JobStatusMsg, waited,
                )

                if status == JOB_STATUS_DONE:
                    urls = response.ResultImage
                    if urls:
                        return urls
                    logger.warning("  → 任务完成但结果 URL 为空 [%s]", job_id)
                    return None

                if status == JOB_STATUS_FAILED:
                    logger.error(
                        "  → 任务失败 [%s]: %s - %s",
                        job_id, response.JobErrorCode, response.JobErrorMsg,
                    )
                    return None

                # 状态 1/2：继续等待

            except TencentCloudSDKException as exc:
                logger.error("查询任务失败 [%s]: %s", job_id, exc)
                return None

        logger.error("  → 任务超时（等待超过 %ds）[%s]", POLL_MAX_WAIT_SECONDS, job_id)
        return None

    def _process_single_image(
        self,
        image_path: Path,
        output_path: Path,
        prompt: str,
    ) -> bool:
        """
        处理单张图片：提交任务 → 轮询等待 → 下载结果。

        Returns:
            True = 成功，False = 失败
        """
        # Step 1：提交混元生图 3.0 图生图任务
        job_id = self._submit_image_to_image_job(image_path, prompt)
        if not job_id:
            return False

        # Step 2：轮询等待任务完成
        result_urls = self._poll_job_result(job_id)
        if not result_urls:
            return False

        # Step 3：下载第一张结果图片
        result_url = result_urls[0]
        logger.info("  → 下载结果图片: %s", result_url[:80] + "...")
        if download_image(result_url, output_path):
            return True

        return False

    def batch_process(
        self,
        input_dir: str | Path,
        delay_seconds: float = 2.0,
    ) -> str:
        """
        批量处理目录中的所有图片。

        流程（每张图）：
        1. LangChain Chain 动态生成 Prompt（调用混元 LLM）
        2. SubmitTextToImageJob 提交图生图任务（混元生图 3.0）
        3. 轮询 QueryTextToImageJob 等待任务完成
        4. 下载结果图片，保存到带时间日期的子目录

        Args:
            input_dir:      输入图片目录
            delay_seconds:  每张处理完后的等待时间（秒），避免触发限速

        Returns:
            输出目录路径字符串
        """
        input_dir = Path(input_dir)
        if not input_dir.is_dir():
            raise NotADirectoryError(f"目录不存在: {input_dir}")

        image_files = sorted(
            p for p in input_dir.iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
        )

        if not image_files:
            raise ValueError(f"目录 [{input_dir}] 中未找到支持的图片文件")

        total = len(image_files)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = input_dir / f"ai_modified_{timestamp}_{total}"
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("=" * 60)
        logger.info("开始批量 AI 图片处理（混元生图 3.0 图生图）")
        logger.info("输入目录：%s", input_dir)
        logger.info("输出目录：%s", output_dir)
        logger.info("图片总数：%d 张", total)
        logger.info("=" * 60)

        success_count = 0
        fail_count = 0

        for idx, image_path in enumerate(image_files, start=1):
            logger.info("[%d/%d] 正在处理：%s", idx, total, image_path.name)

            # Step 1：LangChain Chain 动态生成 Prompt
            logger.info("  → LangChain 生成 Prompt 中...")
            prompt = self._generate_prompt(image_path.name)
            logger.info("  → Prompt: %s", prompt[:60] + "..." if len(prompt) > 60 else prompt)

            # Step 2-4：提交任务 → 轮询 → 下载结果
            output_path = output_dir / (image_path.stem + ".jpg")
            if self._process_single_image(image_path, output_path, prompt):
                success_count += 1
                logger.info("  [成功] 已保存 -> %s", output_path.name)
            else:
                fail_count += 1
                logger.warning("  [失败] 跳过：%s", image_path.name)

            if idx < total:
                time.sleep(delay_seconds)

        logger.info("=" * 60)
        logger.info("处理完成！成功：%d 张，失败：%d 张", success_count, fail_count)
        logger.info("结果目录：%s", output_dir)
        logger.info("=" * 60)

        return str(output_dir)


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

def main() -> None:
    """从 .env 文件读取密钥，执行批量图片处理。"""

    secret_id = os.environ.get("TENCENT_SECRET_ID", "").strip()
    secret_key = os.environ.get("TENCENT_SECRET_KEY", "").strip()

    if not secret_id or not secret_key:
        logger.error(
            "缺少腾讯云 API 密钥！\n"
            "请在项目根目录的 .env 文件中添加：\n"
            "  TENCENT_SECRET_ID=<您的 SecretId>\n"
            "  TENCENT_SECRET_KEY=<您的 SecretKey>"
        )
        sys.exit(1)

    # ★ 修改此变量指定要处理的图片目录
    input_dir = "pics"

    processor = HunyuanImageProcessor(
        secret_id=secret_id,
        secret_key=secret_key,
        resolution="1024:1024",   # 输出分辨率，可按需调整
    )

    try:
        output_dir = processor.batch_process(
            input_dir=input_dir,
            delay_seconds=2.0,    # 异步任务无需太频繁，2秒间隔即可
        )
        print(f"\n全部处理完成！结果保存在：{output_dir}")
    except (NotADirectoryError, ValueError) as exc:
        logger.error("目录错误：%s", exc)
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("用户中断处理")
        sys.exit(0)


if __name__ == "__main__":
    main()

