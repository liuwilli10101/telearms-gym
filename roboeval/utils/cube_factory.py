import pathlib, re, string, random
from roboeval.utils.generate_ordered_cubes import ordered_cubes   
from typing import Union, List

# Location pattern that your generator writes to
# Find RoboEval root directory dynamically
_THIS_FILE = pathlib.Path(__file__).resolve()
ROBOEVAL_ROOT = _THIS_FILE.parent.parent.parent  # roboeval/utils/cube_factory.py -> RoboEval/
XML_DIR = ROBOEVAL_ROOT / "roboeval/envs/xmls/props/cube/ordered_cubes"

_png_re  = re.compile(r'file="([^"]+)\.png"')

def _labels_from_xml(xml_path: pathlib.Path) -> list[str]:
    """Parse the six texture-filenames (labels) in the order [+X −X +Y −Y +Z −Z]."""
    txt = xml_path.read_text()
    return _png_re.findall(txt)[:6]         # returns ["P","S","C","I","E","G"]

def ensure_cubes(word: str,
                 universe: Union[str, List[str]] | None = None) -> dict[str, dict]:
    """
    Make sure *all* cubes for `word` exist.
    
    IMPORTANT: Each character in `word` becomes ONE cube!
    - "ABC" creates 3 cubes (A, B, C)
    - "123" creates 3 cubes (1, 2, 3)
    - "123456789" is the MAX for numeric sequences (9 cubes)
    
    ⚠️ Multi-digit numbers like "10", "11" are NOT supported!
    Each digit becomes a separate cube. Use letters A-Z for 10+ items.
    
    Returns
    -------
    {letter:
        {"xml":   Path  ,          # path to cube_*.xml
         "labels": list[str]}      # six labels in geom order
    }
    """
    word      = word.upper()
    universe  = universe or string.ascii_uppercase
    
    # Validation: Check for potential multi-digit number confusion
    # If word contains sequence like "10", "11", "12" user likely wants multi-digit labels
    if re.search(r'(10|11|12|13|14|15|16|17|18|19|20)', word):
        import warnings
        warnings.warn(
            f"\n⚠️ WARNING: ensure_cubes() detected multi-digit pattern in '{word}'.\n"
            f"Each CHARACTER becomes ONE cube - '10' becomes cubes '1' and '0', NOT a single '10' cube!\n"
            f"For numeric ordering beyond 9 items, use LETTERS (A-Z) instead.\n"
            f"Example: Use 'ABCDEFGHIJ' for 10 cubes, not '12345678910'.",
            UserWarning
        )
    
    word_dir  = XML_DIR / word
    word_dir.mkdir(parents=True, exist_ok=True)

    # 1) generate only when *one* cube is missing
    missing = [c for c in word if not (word_dir / f"cube_{c}.xml").exists()]
    if missing:
        # call your generator once – it overwrites nothing, just creates the missing XMLs
        ordered_cubes(word, universe=universe)

    # 2) build dictionary
    out: dict[str, dict] = {}
    for ch in word:
        xml_path = word_dir / f"cube_{ch}.xml"
        out[ch]  = {"xml": xml_path,
                    "labels": _labels_from_xml(xml_path)}
    return out
