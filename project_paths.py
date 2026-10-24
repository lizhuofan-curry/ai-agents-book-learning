"""读取迁移前路径；原始清单、模型和实验记录保持原字节与哈希。"""

from pathlib import Path, PureWindowsPath


PROJECT_ROOT = Path(__file__).resolve().parent


def project_path(value, *, root=PROJECT_ROOT):
    """将项目内的 Windows/相对路径解析到当前 checkout，拒绝越界。"""
    root = Path(root).resolve()
    text = str(value).replace('\\', '/')
    windows = PureWindowsPath(text)
    if windows.drive:
        if not windows.is_absolute():
            raise ValueError('不支持相对盘符路径：' + text)
        parts = windows.parts[1:]
        matches = [i for i, part in enumerate(parts) if part.casefold() == root.name.casefold()]
        if len(matches) != 1:
            raise ValueError('旧绝对路径必须明确属于本项目：' + text)
        candidate = root.joinpath(*parts[matches[0] + 1:])
    else:
        path = Path(text)
        candidate = path if path.is_absolute() else root / path
    candidate = candidate.resolve()
    if not candidate.is_relative_to(root):
        raise ValueError('路径超出项目目录：' + text)
    return candidate


def path_key_value(mapping, relative):
    """兼容历史 source_hashes 中的反斜杠键，拒绝歧义。"""
    target = str(relative).replace('\\', '/')
    matches = [value for key, value in mapping.items() if str(key).replace('\\', '/') == target]
    if len(matches) != 1:
        raise ValueError('路径索引缺失或重复：' + target)
    return matches[0]
