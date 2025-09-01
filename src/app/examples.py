from PIL import Image

make_dict = lambda x : { 'background': Image.open(x).convert("RGBA"), 'layers': [Image.new("RGBA", Image.open(x).size, (255, 255, 255, 0))], 'composite': Image.open(x).convert("RGBA") }
    
null_dict = {
    'background': None,
    'composite': None,
    'layers': []
}

IMG_REF = [
    ## pos-aware: precise mask
    "assets/gradio/pos_aware/001/img_ref.png",
    "assets/gradio/pos_aware/002/img_ref.png", 
    ## pos-aware: User-drawn mask mask
    "assets/gradio/pos_aware/003/img_ref.png",
    "assets/gradio/pos_aware/004/img_ref.png",
    "assets/gradio/pos_aware/005/img_ref.png",
    ## pos-free
    "assets/gradio/pos_free/001/img_ref.png",
    "assets/gradio/pos_free/002/img_ref.png",
    "assets/gradio/pos_free/003/img_ref.png",
    "assets/gradio/pos_free/004/img_ref.png",
    ]

IMG_TGT1 = [
    ## pos-aware: precise mask
    "assets/gradio/pos_aware/001/img_target.png", 
    "assets/gradio/pos_aware/002/img_target.png", 
    ## pos-aware: User-drawn mask mask
    None,
    None,
    None,
    ## pos-free
    "assets/gradio/pos_free/001/img_target.png", 
    "assets/gradio/pos_free/002/img_target.png",
    "assets/gradio/pos_free/003/img_target.png",
    "assets/gradio/pos_free/004/img_target.png",
    ]

IMG_TGT2 = [
    ## pos-aware: precise mask
    null_dict, 
    null_dict, 
    ## pos-aware: User-drawn mask mask
    make_dict("assets/gradio/pos_aware/003/img_target.png"), 
    make_dict("assets/gradio/pos_aware/004/img_target.png"),
    make_dict("assets/gradio/pos_aware/005/img_target.png"),
    ## pos-free
    null_dict,
    null_dict,
    null_dict,
    null_dict,
    ]

MASK_TGT = [
    ## pos-aware: precise mask
    "assets/gradio/pos_aware/001/mask_target.png", 
    "assets/gradio/pos_aware/002/mask_target.png", 
    ## pos-aware: User-drawn mask mask
    "assets/gradio/pos_aware/003/mask_target.png", 
    "assets/gradio/pos_aware/004/mask_target.png",
    "assets/gradio/pos_aware/005/mask_target.png",
    ## pos-free
    "assets/gradio/pos_free/001/mask_target.png",
    "assets/gradio/pos_free/002/mask_target.png", 
    "assets/gradio/pos_free/003/mask_target.png",
    "assets/gradio/pos_free/004/mask_target.png",
    ]

CUSTOM_MODE = [
    ## pos-aware
    "Position-aware",  
    "Position-aware", 
    "Position-aware", 
    "Position-aware",
    "Position-aware",
    ## pos-free
    "Position-free",
    "Position-free",
    "Position-free",
    "Position-free",
    ]

INPUT_MASK_MODE = [
    ## pos-aware: precise mask
    "Precise mask", 
    "Precise mask", 
    ## pos-aware: User-drawn mask mask
    "User-drawn mask", 
    "User-drawn mask",
    "User-drawn mask",
    ## pos-free
    "Precise mask",
    "Precise mask",
    "Precise mask",
    "Precise mask",
    ]

SEG_REF_MODE = [
    ## pos-aware
    "Full Ref", 
    "Full Ref", 
    "Full Ref", 
    "Full Ref", 
    "Full Ref",
    ## pos-free
    "Full Ref",
    "Full Ref",
    "Full Ref",
    "Full Ref",
    ]

PROMPTS = [
    ## pos-aware: precise mask
    "", 
    "", 
    ## pos-aware: User-drawn mask mask
    "A delicate necklace with a mother-of-pearl clover pendant hangs gracefully around the neck of a woman dressed in a black pinstripe blazer.",
    "",
    "",
    ## pos-free
    "TThe charming, soft plush toy is joyfully wandering through a lush, dense jungle, surrounded by vibrant green foliage and towering trees.",
    "A bright yellow alarm clock sits on a wooden desk next to a stack of books in a cozy, sunlit room.",
    "A Lego figure dressed in a vibrant chicken costume, leaning against a wooden chair, surrounded by lush green grass and blooming flowers.",
    "The crocheted gingerbread man is perched on a tree branch in a dense forest, with sunlight filtering through the leaves, casting dappled shadows around him."
    ]

IMG_GEN = [
    ## pos-aware: precise mask
    "assets/gradio/pos_aware/001/img_gen.png", 
    "assets/gradio/pos_aware/002/img_gen.png", 
    ## pos-aware: User-drawn mask mask
    "assets/gradio/pos_aware/003/img_gen.png",
    "assets/gradio/pos_aware/004/img_gen.png",
    "assets/gradio/pos_aware/005/img_gen.png",
    ## pos-free
    "assets/gradio/pos_free/001/img_gen.png",
    "assets/gradio/pos_free/002/img_gen.png",
    "assets/gradio/pos_free/003/img_gen.png",
    "assets/gradio/pos_free/004/img_gen.png",
    ]

SEED = [
    ## pos-aware
    97175498,
    513097943, 
    346969695,
    1172525388,
    268683460,
    ## pos-free
    2126677963,
    418898253,
    2126677963,
    2126677963
    ]

TRUE_GS = [
    # pos-aware
    1,
    1,
    1,
    1,
    1,
    # pos-free
    3,
    3,
    3,
    3,
]

NUM_STEPS = [
    ## pos-aware
    32,
    32,
    32,
    32,
    32,
    ## pos-free
    20,
    20,
    20,
    20,
]

GUIDANCE = [
    ## pos-aware
    40,
    48,
    40,
    48,
    48,
    ## pos-free
    40,
    40,
    40,
    40,
]

GRADIO_EXAMPLES = [
    [IMG_REF[0], IMG_TGT1[0], IMG_TGT2[0], CUSTOM_MODE[0], INPUT_MASK_MODE[0], SEG_REF_MODE[0], PROMPTS[0], SEED[0], TRUE_GS[0], '0', NUM_STEPS[0], GUIDANCE[0]],
    [IMG_REF[1], IMG_TGT1[1], IMG_TGT2[1], CUSTOM_MODE[1], INPUT_MASK_MODE[1], SEG_REF_MODE[1], PROMPTS[1], SEED[1], TRUE_GS[1], '1', NUM_STEPS[1], GUIDANCE[1]],
    [IMG_REF[2], IMG_TGT1[2], IMG_TGT2[2], CUSTOM_MODE[2], INPUT_MASK_MODE[2], SEG_REF_MODE[2], PROMPTS[2], SEED[2], TRUE_GS[2], '2', NUM_STEPS[2], GUIDANCE[2]],
    [IMG_REF[3], IMG_TGT1[3], IMG_TGT2[3], CUSTOM_MODE[3], INPUT_MASK_MODE[3], SEG_REF_MODE[3], PROMPTS[3], SEED[3], TRUE_GS[3], '3', NUM_STEPS[3], GUIDANCE[3]],
    [IMG_REF[4], IMG_TGT1[4], IMG_TGT2[4], CUSTOM_MODE[4], INPUT_MASK_MODE[4], SEG_REF_MODE[4], PROMPTS[4], SEED[4], TRUE_GS[4], '4', NUM_STEPS[4], GUIDANCE[4]],
    [IMG_REF[5], IMG_TGT1[5], IMG_TGT2[5], CUSTOM_MODE[5], INPUT_MASK_MODE[5], SEG_REF_MODE[5], PROMPTS[5], SEED[5], TRUE_GS[5], '5', NUM_STEPS[5], GUIDANCE[5]],
    [IMG_REF[6], IMG_TGT1[6], IMG_TGT2[6], CUSTOM_MODE[6], INPUT_MASK_MODE[6], SEG_REF_MODE[6], PROMPTS[6], SEED[6], TRUE_GS[6], '6', NUM_STEPS[6], GUIDANCE[6]],
    [IMG_REF[7], IMG_TGT1[7], IMG_TGT2[7], CUSTOM_MODE[7], INPUT_MASK_MODE[7], SEG_REF_MODE[7], PROMPTS[7], SEED[7], TRUE_GS[7], '7', NUM_STEPS[7], GUIDANCE[7]],
    [IMG_REF[8], IMG_TGT1[8], IMG_TGT2[8], CUSTOM_MODE[8], INPUT_MASK_MODE[8], SEG_REF_MODE[8], PROMPTS[8], SEED[8], TRUE_GS[8], '8', NUM_STEPS[8], GUIDANCE[8]],
]