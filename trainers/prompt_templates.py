"""
Prompt 模板（用于 BiomedCoOp 的 teacher prompt 构造）。

该文件缺失会导致 `trainers/BiomedCoOp/*` import 失败：
`from trainers.prompt_templates import biomedcoop_template_at`
"""

IMAGENET_TEMPLATES_SELECT = [
    "itap of a {}.",
    "a bad photo of the {}.",
    "a origami {}.",
    "a photo of the large {}.",
    "a {} in a video game.",
    "art of the {}.",
    "a photo of the small {}.",
]


def biomedcoop_template_at(classname: str, i: int) -> str:
    """返回第 i 个 prompt（循环取模）。"""
    name = str(classname).replace("_", " ").strip()
    return IMAGENET_TEMPLATES_SELECT[int(i) % len(IMAGENET_TEMPLATES_SELECT)].format(name)

