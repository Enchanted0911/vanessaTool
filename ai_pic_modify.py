"""
ai_pic_modify.py
================
批量 AI 图片处理工具：
- 使用 LangChain ChatOpenAI（对接腾讯混元 OpenAI 兼容接口）生成图像编辑指令
- 使用腾讯云 aiart ImageToImage 接口执行去水印 + 改模特姿势
- 保留衣服细节，保证输出高清清晰度
- 结果保存到带时间日期的子目录

使用方式：
    uv run python ai_pic_modify.py

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

# 负向提示词
NEGATIVE_PROMPT = (
    "水印, 文字, logo, 文字叠加, 模糊, 低分辨率, 低画质, 噪点, 失真, "
    "衣服改变, 服装变形, 颜色改变, 款式改变, 图案丢失, 面料改变"
)

# ---------------------------------------------------------------------------
# LangChain Chain：Prompt → 混元 LLM → 解析
# ---------------------------------------------------------------------------

_SYSTEM_MSG = """\
你是专业的时尚电商图片处理助手。
根据用户提供的图片文件名，生成一段用于腾讯云 ImageToImage 接口的图像编辑 Prompt。
严格要求：
1. 去除图片中所有水印、文字覆盖和 logo，还原干净背景
2. 改变模特站姿，使其更自然优雅（侧身展示、手部姿势自然变化）
3. 严格保持模特所穿衣服的颜色、款式、面料、图案和版型完全不变
4. 整体画质清晰，细节丰富，专业商品展示风格
只输出 Prompt 本身，中文，不超过 256 字，不要有任何额外解释。"""

_HUMAN_MSG = "图片文件名：{filename}"

# 构建 LangChain 链：ChatPromptTemplate | ChatOpenAI(混元) | StrOutputParser
def build_prompt_chain(api_key: str) -> Runnable:
    """
    构建 LangChain Prompt 生成链。

    使用 ChatOpenAI 接入腾讯混元 OpenAI 兼容接口，
    通过 | 管道运算符组成：prompt_template | llm | parser 的标准 LCEL 链。

    Args:
        api_key: 腾讯混元 API Key（从 .env 读取的 TENCENT_SECRET_KEY）

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


def save_base64_image(b64_data: str, output_path: Path) -> None:
    """将 Base64 图片数据解码并写入文件。"""
    with open(output_path, "wb") as f:
        f.write(base64.b64decode(b64_data))


# ---------------------------------------------------------------------------
# 核心处理器
# ---------------------------------------------------------------------------

class HunyuanImageProcessor:
    """
    腾讯混元图片处理器。

    - 使用 LangChain ChatOpenAI（混元兼容接口）动态生成图像编辑 Prompt
    - 使用腾讯云 aiart ImageToImage 接口完成去水印 + 改姿势
    - 开启画质增强，保证输出清晰度
    """

    def __init__(
        self,
        secret_id: str,
        secret_key: str,
        strength: float = 0.65,
        enhance_image: int = 1,
        restore_face: int = 2,
    ) -> None:
        """
        Args:
            secret_id:      腾讯云 SecretId（用于 aiart SDK 鉴权）
            secret_key:     腾讯云 SecretKey（同时作为混元 OpenAI 兼容接口的 API Key）
            strength:       ImageToImage 生成自由度 (0, 1]，推荐 0.6-0.8
            enhance_image:  画质增强开关，1=开启，0=关闭
            restore_face:   面部细节优化数量上限，0-6
        """
        # aiart 图像生成客户端（用 SecretId + SecretKey 鉴权）
        cred = credential.Credential(secret_id, secret_key)
        self.aiart_client = aiart_client.AiartClient(cred, "ap-guangzhou")

        api_key = os.environ.get("API_KEY", "").strip()
        # LangChain Chain（用 SecretKey 作为混元 OpenAI 兼容接口的 API Key）
        self.prompt_chain = build_prompt_chain(api_key=api_key)
        # self.prompt_chain = build_prompt_chain(api_key='sk-Cey67taEEG2Jjx2fdPI9HtdYsLwLw1zsRwva0iTdrPC6guCb')
        self.strength = strength
        self.enhance_image = enhance_image
        self.restore_face = restore_face

    def _generate_prompt(self, filename: str) -> str:
        """
        调用 LangChain Chain 动态生成当前图片的编辑 Prompt。

        链路：ChatPromptTemplate | ChatOpenAI(混元) | StrOutputParser
        """
        try:
            prompt = self.prompt_chain.invoke({"filename": filename})
            logger.info("LangChain 生成 Prompt [%s]: %s", filename, prompt)
            return prompt.strip()
        except Exception as exc:
            logger.warning("LangChain 生成 Prompt 失败，使用默认 Prompt。原因：%s", exc)
            return (
                "高清时尚模特图，去除图片中所有水印、文字覆盖和logo，"
                "修改模特的站姿使其更自然优雅，"
                "严格保持模特所穿衣服的颜色、款式、面料、图案和版型完全不变，"
                "整体画质清晰，细节丰富，专业商品展示风格"
            )

    def _call_image_to_image(
        self,
        image_path: Path,
        output_path: Path,
        prompt: str,
    ) -> bool:
        """
        调用腾讯云 aiart ImageToImage 接口处理单张图片。

        Returns:
            True = 成功，False = 失败
        """
        try:
            req = models.ImageToImageRequest()
            req.InputImage = image_to_base64(image_path)
            req.Prompt = prompt
            req.NegativePrompt = NEGATIVE_PROMPT
            req.Strength = self.strength
            req.EnhanceImage = self.enhance_image
            req.RestoreFace = self.restore_face
            req.LogoAdd = 0            # 不添加 AI 水印标识
            req.RspImgType = "base64"

            # 保持原始分辨率（长边最高 2000px）
            result_config = models.ResultConfig()
            result_config.Resolution = "origin"
            req.ResultConfig = result_config

            response = self.aiart_client.ImageToImage(req)

            if response.ResultImage:
                save_base64_image(response.ResultImage, output_path)
                return True

            logger.warning("图片 [%s] API 返回结果为空", image_path.name)
            return False

        except TencentCloudSDKException as exc:
            logger.error("aiart API 调用失败 [%s]: %s", image_path.name, exc)
            return False
        except Exception as exc:
            logger.error("处理图片 [%s] 异常: %s", image_path.name, exc)
            return False

    def batch_process(
        self,
        input_dir: str | Path,
        delay_seconds: float = 1.5,
    ) -> str:
        """
        批量处理目录中的所有图片。

        流程（每张图）：
        1. LangChain Chain 动态生成 Prompt（调用混元 LLM）
        2. aiart ImageToImage 接口执行图片处理
        3. 保存结果到带时间日期的子目录

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
        logger.info("开始批量 AI 图片处理（去水印 + 改姿势 + 保留衣服）")
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

            # Step 2：aiart ImageToImage 处理图片
            output_path = output_dir / (image_path.stem + ".jpg")
            if self._call_image_to_image(image_path, output_path, prompt):
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
        strength=0.65,    # 适中自由度：改变姿势同时保留衣服细节
        enhance_image=1,  # 开启画质增强，保证清晰度
        restore_face=2,   # 优化最多 2 个人脸细节
    )

    try:
        output_dir = processor.batch_process(
            input_dir=input_dir,
            delay_seconds=1.5,
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

