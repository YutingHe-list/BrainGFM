"""只提供 LPIPS 用到的仓库根目录。不在导入时改写编译缓存。"""
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
