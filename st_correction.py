import opencc


CONVERTER_T2S = opencc.OpenCC("t2s")
CONVERTER_S2T = opencc.OpenCC("s2t")

def do_st_corrections(text: str) -> str:
    simplified = CONVERTER_T2S.convert(text)

    return CONVERTER_S2T.convert(simplified)