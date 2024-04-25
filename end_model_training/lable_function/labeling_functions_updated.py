import re
from snorkel.labeling import labeling_function
ABSTAIN = -1

# TODO: abreviation
# Check each class's accuracy

@labeling_function()
def entity_label_1(x):
    # Apply the regex ( |^)(name|name)[^\w]* (\w+ ){0,1}(a)[^\w]*( |$) to x.text
    # If the regex matches, return 1 else return ABSTAIN
    return 1 if re.search(r"( |^)(name|name)[^\w]* (\w+ ){0,1}(a)[^\w]*( |$)", x.text.lower()) else ABSTAIN


@labeling_function()
def description_label_2(x):
    # Apply the regex ( |^)(how|how)[^\w]* (\w+ ){0,1}(does|to|can|should|would|could|will|do|do)[^\w]*( |$) to x.text
    # If the regex matches, return 1 else return ABSTAIN
    return 1 if re.search(r"( |^)(how|how)[^\w]* (\w+ ){0,1}(does|to|can|should|would|could|will|do|do)[^\w]*( |$)", x.text.lower()) else ABSTAIN


@labeling_function()
def description_label_3(x):
    # Apply the regex ( |^)(what|what)[^\w]* (\w+ ){0,1}(is|is)[^\w]* ([^\s]+ )*(hypertension|hypertension)[^\w]*( |$) to x.text
    # If the regex matches, return 1 else return ABSTAIN
    return 1 if re.search(r"( |^)(what|what)[^\w]* (\w+ ){0,1}(is|is)[^\w]* ([^\s]+ )*(hypertension|hypertension)[^\w]*( |$)", x.text.lower()) else ABSTAIN


@labeling_function()
def location_label_4(x):
    # Apply the regex ( |^)(which|where|what|what)[^\w]* ([^\s]+ )*(near|close to|far|around|surrounds|surrounds)[^\w]*( |$) to x.text
    # If the regex matches, return 1 else return ABSTAIN
    return 1 if re.search(r"( |^)(which|where|what|what)[^\w]* ([^\s]+ )*(near|close to|far|around|surrounds|surrounds)[^\w]*( |$)", x.text.lower()) else ABSTAIN


@labeling_function()
def human_label_5(x):
    # Apply the regex ( |^)(which|who|what|what)[^\w]* ([^\s]+ )*(person|man|woman|human|poet|poet)[^\w]*( |$) to x.text
    # If the regex matches, return 1 else return ABSTAIN
    return 1 if re.search(r"( |^)(which|who|what|what)[^\w]* ([^\s]+ )*(person|man|woman|human|poet|poet)[^\w]*( |$)", x.text.lower()) else ABSTAIN


@labeling_function()
def entity_label_6(x):
    # Apply the regex ( |^)(what|what)[^\w]* (\w+ ){0,1}(.*er|fastener|fastener)[^\w]* ([^\s]+ )*(played|play|run|study |studied|patent|patent)[^\w]*( |$) to x.text
    # If the regex matches, return 1 else return ABSTAIN
    return 1 if re.search(r"( |^)(what|what)[^\w]* (\w+ ){0,1}(.*er|fastener|fastener)[^\w]* ([^\s]+ )*(played|play|run|study |studied|patent|patent)[^\w]*( |$)", x.text.lower()) else ABSTAIN


@labeling_function()
def description_label_7(x):
    # Apply the regex ( |^)(how|how)[^\w]* (\w+ ){0,1}(do|do)[^\w]* (\w+ ){0,1}(you|you)[^\w]*( |$) to x.text
    # If the regex matches, return 1 else return ABSTAIN
    return 1 if re.search(r"( |^)(how|how)[^\w]* (\w+ ){0,1}(do|do)[^\w]* (\w+ ){0,1}(you|you)[^\w]*( |$)", x.text.lower()) else ABSTAIN


@labeling_function()
def human_label_8(x):
    # Apply the regex ( |^)(who|what|what)[^\w]* (\w+ ){0,1}(person|man|woman|human|president|president)[^\w]*( |$) to x.text
    # If the regex matches, return 1 else return ABSTAIN
    return 1 if re.search(r"( |^)(who|what|what)[^\w]* (\w+ ){0,1}(person|man|woman|human|president|president)[^\w]*( |$)", x.text.lower()) else ABSTAIN


@labeling_function()
def numeric_label_9(x):
    # Apply the regex ( |^)(how|how)[^\w]* (\w+ ){0,1}(much|many|many)[^\w]*( |$) to x.text
    # If the regex matches, return 1 else return ABSTAIN
    return 1 if re.search(r"( |^)(how|how)[^\w]* (\w+ ){0,1}(much|many|many)[^\w]*( |$)", x.text.lower()) else ABSTAIN


@labeling_function()
def location_label_10(x):
    # Apply the regex ( |^)(which|what|where|where)[^\w]* ([^\s]+ )*(situated|located|located)[^\w]*( |$) to x.text
    # If the regex matches, return 1 else return ABSTAIN
    return 1 if re.search(r"( |^)(which|what|where|where)[^\w]* ([^\s]+ )*(situated|located|located)[^\w]*( |$)", x.text.lower()) else ABSTAIN


@labeling_function()
def human_label_11(x):
    # Apply the regex ( |^)(who|who)[^\w]* ([^\s]+ )*(man|woman|human|person|person)[^\w]*( |$) to x.text
    # If the regex matches, return 1 else return ABSTAIN
    return 1 if re.search(r"( |^)(who|who)[^\w]* ([^\s]+ )*(man|woman|human|person|person)[^\w]*( |$)", x.text.lower()) else ABSTAIN


@labeling_function()
def entity_label_12(x):
    # Apply the regex ( |^)(which|what|what)[^\w]* ([^\s]+ )*(team|group|groups|teams|teams)[^\w]*( |$) to x.text
    # If the regex matches, return 1 else return ABSTAIN
    return 1 if re.search(r"( |^)(which|what|what)[^\w]* ([^\s]+ )*(team|group|groups|teams|teams)[^\w]*( |$)", x.text.lower()) else ABSTAIN


@labeling_function()
def location_label_13(x):
    # Apply the regex ( |^)(where|where)[^\w]* ([^\s]+ )*(stand|stand)[^\w]*( |$) to x.text
    # If the regex matches, return 1 else return ABSTAIN
    return 1 if re.search(r"( |^)(where|where)[^\w]* ([^\s]+ )*(stand|stand)[^\w]*( |$)", x.text.lower()) else ABSTAIN


@labeling_function()
def description_label_14(x):
    # Apply the regex ( |^)(what|what)[^\w]* ([^\s]+ )*(mean|meant|meant)[^\w]*( |$) to x.text
    # If the regex matches, return 1 else return ABSTAIN
    return 1 if re.search(r"( |^)(what|what)[^\w]* ([^\s]+ )*(mean|meant|meant)[^\w]*( |$)", x.text.lower()) else ABSTAIN


@labeling_function()
def entity_label_15(x):
    # Apply the regex ( |^)(what|what)[^\w]* (\w+ ){0,1}(kind|kind)[^\w]*( |$) to x.text
    # If the regex matches, return 1 else return ABSTAIN
    return 1 if re.search(r"( |^)(what|what)[^\w]* (\w+ ){0,1}(kind|kind)[^\w]*( |$)", x.text.lower()) else ABSTAIN


@labeling_function()
def numeric_label_16(x):
    # Apply the regex ( |^)(what|what)[^\w]* (\w+ ){0,1}(amount|number|percentage|percentage)[^\w]*( |$) to x.text
    # If the regex matches, return 1 else return ABSTAIN
    return 1 if re.search(r"( |^)(what|what)[^\w]* (\w+ ){0,1}(amount|number|percentage|percentage)[^\w]*( |$)", x.text.lower()) else ABSTAIN


@labeling_function()
def location_label_17(x):
    # Apply the regex ( |^)(capital|capital)[^\w]* (\w+ ){0,1}(of|of)[^\w]*( |$) to x.text
    # If the regex matches, return 1 else return ABSTAIN
    return 1 if re.search(r"( |^)(capital|capital)[^\w]* (\w+ ){0,1}(of|of)[^\w]*( |$)", x.text.lower()) else ABSTAIN


@labeling_function()
def description_label_18(x):
    # Apply the regex ( |^)(why|why)[^\w]* (\w+ ){0,1}(does|should |shall|could|would|will|can|do|do)[^\w]*( |$) to x.text
    # If the regex matches, return 1 else return ABSTAIN
    return 1 if re.search(r"( |^)(why|why)[^\w]* (\w+ ){0,1}(does|should |shall|could|would|will|can|do|do)[^\w]*( |$)", x.text.lower()) else ABSTAIN


@labeling_function()
def entity_label_19(x):
    # Apply the regex ( |^)(composed|made|made)[^\w]* (\w+ ){0,1}(from|through|using|by|of|of)[^\w]*( |$) to x.text
    # If the regex matches, return 1 else return ABSTAIN
    return 1 if re.search(r"( |^)(composed|made|made)[^\w]* (\w+ ){0,1}(from|through|using|by|of|of)[^\w]*( |$)", x.text.lower()) else ABSTAIN


@labeling_function()
def location_label_20(x):
    # Apply the regex ( |^)(where|which|what|what)[^\w]* (\w+ ){0,1}(island|island)[^\w]*( |$) to x.text
    # If the regex matches, return 1 else return ABSTAIN
    return 1 if re.search(r"( |^)(where|which|what|what)[^\w]* (\w+ ){0,1}(island|island)[^\w]*( |$)", x.text.lower()) else ABSTAIN


@labeling_function()
def human_label_21(x):
    # Apply the regex ( |^)(who|who)[^\w]* (\w+ ){0,1}(owner|leads|governs|pays|owns|owns)[^\w]*( |$) to x.text
    # If the regex matches, return 1 else return ABSTAIN
    return 1 if re.search(r"( |^)(who|who)[^\w]* (\w+ ){0,1}(owner|leads|governs|pays|owns|owns)[^\w]*( |$)", x.text.lower()) else ABSTAIN


@labeling_function()
def description_label_22(x):
    # Apply the regex ( |^)(what|what)[^\w]* (\w+ ){0,1}(is|is)[^\w]* (\w+ ){0,1}(tetrinet|tetrinet)[^\w]*( |$) to x.text
    # If the regex matches, return 1 else return ABSTAIN
    return 1 if re.search(r"( |^)(what|what)[^\w]* (\w+ ){0,1}(is|is)[^\w]* (\w+ ){0,1}(tetrinet|tetrinet)[^\w]*( |$)", x.text.lower()) else ABSTAIN


@labeling_function()
def human_label_23(x):
    # Apply the regex ( |^)(who|who)[^\w]* (\w+ ){0,1}(found|discovered|made|built|build|invented|invented)[^\w]*( |$) to x.text
    # If the regex matches, return 1 else return ABSTAIN
    return 1 if re.search(r"( |^)(who|who)[^\w]* (\w+ ){0,1}(found|discovered|made|built|build|invented|invented)[^\w]*( |$)", x.text.lower()) else ABSTAIN


@labeling_function()
def entity_label_24(x):
    # Apply the regex ( |^)(what|what)[^\w]* ([^\s]+ )*(called|called)[^\w]*( |$) to x.text
    # If the regex matches, return 1 else return ABSTAIN
    return 1 if re.search(r"( |^)(what|what)[^\w]* ([^\s]+ )*(called|called)[^\w]*( |$)", x.text.lower()) else ABSTAIN


@labeling_function()
def description_label_25(x):
    # Apply the regex ( |^)(unusual|unusual)[^\w]*( |$) to x.text
    # If the regex matches, return 1 else return ABSTAIN
    return 1 if re.search(r"( |^)(unusual|unusual)[^\w]*( |$)", x.text.lower()) else ABSTAIN


@labeling_function()
def description_label_26(x):
    # Apply the regex ( |^)(what|what)[^\w]* ([^\s]+ )*(origin|origin)[^\w]*( |$) to x.text
    # If the regex matches, return 1 else return ABSTAIN
    return 1 if re.search(r"( |^)(what|what)[^\w]* ([^\s]+ )*(origin|origin)[^\w]*( |$)", x.text.lower()) else ABSTAIN


@labeling_function()
def location_label_27(x):
    # Apply the regex ( |^)(country|country)[^\w]*( |$) to x.text
    # If the regex matches, return 1 else return ABSTAIN
    return 1 if re.search(r"( |^)(country|country)[^\w]*( |$)", x.text.lower()) else ABSTAIN


@labeling_function()
def location_label_28(x):
    # Apply the regex ( |^)(where|where)[^\w]*( |$) to x.text
    # If the regex matches, return 1 else return ABSTAIN
    return 1 if re.search(r"( |^)(where|where)[^\w]*( |$)", x.text.lower()) else ABSTAIN


@labeling_function()
def numeric_label_29(x):
    # Apply the regex ( |^)(which|what|what)[^\w]* ([^\s]+ )*(time|day|month|hours|minute|seconds|year|date|date)[^\w]*( |$) to x.text
    # If the regex matches, return 1 else return ABSTAIN
    return 1 if re.search(r"( |^)(which|what|what)[^\w]* ([^\s]+ )*(time|day|month|hours|minute|seconds|year|date|date)[^\w]*( |$)", x.text.lower()) else ABSTAIN


@labeling_function()
def description_label_30(x):
    # Apply the regex ( |^)(why|why)[^\w]* (\w+ ){0,1}(does|doesn|doesn)[^\w]*( |$) to x.text
    # If the regex matches, return 1 else return ABSTAIN
    return 1 if re.search(r"( |^)(why|why)[^\w]* (\w+ ){0,1}(does|doesn|doesn)[^\w]*( |$)", x.text.lower()) else ABSTAIN


@labeling_function()
def human_label_31(x):
    # Apply the regex ( |^)(queen|king|king)[^\w]*( |$) to x.text
    # If the regex matches, return 1 else return ABSTAIN
    return 1 if re.search(r"( |^)(queen|king|king)[^\w]*( |$)", x.text.lower()) else ABSTAIN


@labeling_function()
def numeric_label_32(x):
    # Apply the regex ( |^)(year|year)[^\w]*( |$) to x.text
    # If the regex matches, return 1 else return ABSTAIN
    return 1 if re.search(r"( |^)(year|year)[^\w]*( |$)", x.text.lower()) else ABSTAIN


@labeling_function()
def entity_label_33(x):
    # Apply the regex ( |^)(novel|novel)[^\w]*( |$) to x.text
    # If the regex matches, return 1 else return ABSTAIN
    return 1 if re.search(r"( |^)(novel|novel)[^\w]*( |$)", x.text.lower()) else ABSTAIN


@labeling_function()
def description_label_34(x):
    # Apply the regex ( |^)(used|used)[^\w]* (\w+ ){0,1}(for|for)[^\w]*( |$) to x.text
    # If the regex matches, return 1 else return ABSTAIN
    return 1 if re.search(r"( |^)(used|used)[^\w]* (\w+ ){0,1}(for|for)[^\w]*( |$)", x.text.lower()) else ABSTAIN


@labeling_function()
def numeric_label_35(x):
    # Apply the regex ( |^)(when|when)[^\w]* (\w+ ){0,1}(did|do|does|was|was)[^\w]*( |$) to x.text
    # If the regex matches, return 1 else return ABSTAIN
    return 1 if re.search(r"( |^)(when|when)[^\w]* (\w+ ){0,1}(did|do|does|was|was)[^\w]*( |$)", x.text.lower()) else ABSTAIN


@labeling_function()
def description_label_36(x):
    # Apply the regex ( |^)(what|what)[^\w]* (\w+ ){0,1}(kind|kind)[^\w]*( |$) to x.text
    # If the regex matches, return 1 else return ABSTAIN
    return 1 if re.search(r"( |^)(what|what)[^\w]* (\w+ ){0,1}(kind|kind)[^\w]*( |$)", x.text.lower()) else ABSTAIN


@labeling_function()
def numeric_label_37(x):
    # Apply the regex ( |^)(how|how)[^\w]* (\w+ ){0,1}(old|far|long|tall|wide|short|small|close|long|long)[^\w]*( |$) to x.text
    # If the regex matches, return 1 else return ABSTAIN
    return 1 if re.search(r"( |^)(how|how)[^\w]* (\w+ ){0,1}(old|far|long|tall|wide|short|small|close|long|long)[^\w]*( |$)", x.text.lower()) else ABSTAIN


@labeling_function()
def numeric_label_38(x):
    # Apply the regex ( |^)(speed|speed)[^\w]*( |$) to x.text
    # If the regex matches, return 1 else return ABSTAIN
    return 1 if re.search(r"( |^)(speed|speed)[^\w]*( |$)", x.text.lower()) else ABSTAIN


@labeling_function()
def abbreviation_label_39(x):
    # Apply the regex ( |^)(abbreviation|abbreviation)[^\w]*( |$) to x.text
    # If the regex matches, return 1 else return ABSTAIN
    return 1 if re.search(r"( |^)(abbreviation|abbreviation)[^\w]*( |$)", x.text.lower()) else ABSTAIN


@labeling_function()
def numeric_label_40(x):
    # Apply the regex ( |^)(how|how)[^\w]* (\w+ ){0,1}(much|many|many)[^\w]*( |$) to x.text
    # If the regex matches, return 1 else return ABSTAIN
    return 1 if re.search(r"( |^)(how|how)[^\w]* (\w+ ){0,1}(much|many|many)[^\w]*( |$)", x.text.lower()) else ABSTAIN


@labeling_function()
def numeric_label_41(x):
    # Apply the regex ( |^)(what|what)[^\w]* ([^\s]+ )*(percentage|share|number|population|population)[^\w]*( |$) to x.text
    # If the regex matches, return 1 else return ABSTAIN
    return 1 if re.search(r"( |^)(what|what)[^\w]* ([^\s]+ )*(percentage|share|number|population|population)[^\w]*( |$)", x.text.lower()) else ABSTAIN


@labeling_function()
def description_label_42(x):
    # Apply the regex ( |^)(explain|describe|what|what)[^\w]*( |$) to x.text
    # If the regex matches, return 1 else return ABSTAIN
    return 1 if re.search(r"( |^)(explain|describe|what|what)[^\w]*( |$)", x.text.lower()) else ABSTAIN


@labeling_function()
def location_label_43(x):
    # Apply the regex ( |^)(located|located)[^\w]*( |$) to x.text
    # If the regex matches, return 1 else return ABSTAIN
    return 1 if re.search(r"( |^)(located|located)[^\w]*( |$)", x.text.lower()) else ABSTAIN


@labeling_function()
def entity_label_44(x):
    # Apply the regex ( |^)(thing|instance|object|object)[^\w]*( |$) to x.text
    # If the regex matches, return 1 else return ABSTAIN
    return 1 if re.search(r"( |^)(thing|instance|object|object)[^\w]*( |$)", x.text.lower()) else ABSTAIN


@labeling_function()
def human_label_45(x):
    # Apply the regex ( |^)(who|who)[^\w]*( |$) to x.text
    # If the regex matches, return 1 else return ABSTAIN
    return 1 if re.search(r"( |^)(who|who)[^\w]*( |$)", x.text.lower()) else ABSTAIN


@labeling_function()
def entity_label_46(x):
    # Apply the regex ( |^)(fear|fear)[^\w]* (\w+ ){0,1}(of|of)[^\w]*( |$) to x.text
    # If the regex matches, return 1 else return ABSTAIN
    return 1 if re.search(r"( |^)(fear|fear)[^\w]* (\w+ ){0,1}(of|of)[^\w]*( |$)", x.text.lower()) else ABSTAIN


@labeling_function()
def description_label_47(x):
    # Apply the regex ( |^)(explain|describe|how|how)[^\w]* (\w+ ){0,1}(can|can)[^\w]*( |$) to x.text
    # If the regex matches, return 1 else return ABSTAIN
    return 1 if re.search(r"( |^)(explain|describe|how|how)[^\w]* (\w+ ){0,1}(can|can)[^\w]*( |$)", x.text.lower()) else ABSTAIN


@labeling_function()
def human_label_48(x):
    # Apply the regex ( |^)(who|who)[^\w]* (\w+ ){0,1}(worked|lived|guarded|watched|played|ate|slept|portrayed|served|served)[^\w]*( |$) to x.text
    # If the regex matches, return 1 else return ABSTAIN
    return 1 if re.search(r"( |^)(who|who)[^\w]* (\w+ ){0,1}(worked|lived|guarded|watched|played|ate|slept|portrayed|served|served)[^\w]*( |$)", x.text.lower()) else ABSTAIN


@labeling_function()
def numeric_label_49(x):
    # Apply the regex ( |^)(what|what)[^\w]* (\w+ ){0,1}(part|division|ratio|percentage|percentage)[^\w]*( |$) to x.text
    # If the regex matches, return 1 else return ABSTAIN
    return 1 if re.search(r"( |^)(what|what)[^\w]* (\w+ ){0,1}(part|division|ratio|percentage|percentage)[^\w]*( |$)", x.text.lower()) else ABSTAIN


@labeling_function()
def description_label_50(x):
    # Apply the regex ( |^)(explain|describe|what|what)[^\w]* ([^\s]+ )*(mean|mean)[^\w]*( |$) to x.text
    # If the regex matches, return 1 else return ABSTAIN
    return 1 if re.search(r"( |^)(explain|describe|what|what)[^\w]* ([^\s]+ )*(mean|mean)[^\w]*( |$)", x.text.lower()) else ABSTAIN


@labeling_function()
def description_label_51(x):
    # Apply the regex ( |^)(what|what)[^\w]* ([^\s]+ )*(demands|take|take)[^\w]*( |$) to x.text
    # If the regex matches, return 1 else return ABSTAIN
    return 1 if re.search(r"( |^)(what|what)[^\w]* ([^\s]+ )*(demands|take|take)[^\w]*( |$)", x.text.lower()) else ABSTAIN


@labeling_function()
def human_label_52(x):
    # Apply the regex ( |^)(who|who)[^\w]* (\w+ ){0,1}(is|will|was|was)[^\w]* ([^\s]+ )*(leader|citizen|captain|nationalist|hero|actor|actress|star|gamer|player|lawyer|president|president)[^\w]*( |$) to x.text
    # If the regex matches, return 1 else return ABSTAIN
    return 1 if re.search(r"( |^)(who|who)[^\w]* (\w+ ){0,1}(is|will|was|was)[^\w]* ([^\s]+ )*(leader|citizen|captain|nationalist|hero|actor|actress|star|gamer|player|lawyer|president|president)[^\w]*( |$)", x.text.lower()) else ABSTAIN


@labeling_function()
def entity_label_53(x):
    # Apply the regex ( |^)(name|name)[^\w]*( |$) to x.text
    # If the regex matches, return 1 else return ABSTAIN
    return 1 if re.search(r"( |^)(name|name)[^\w]*( |$)", x.text.lower()) else ABSTAIN


@labeling_function()
def description_label_54(x):
    # Apply the regex ( |^)(how|what|what)[^\w]* (\w+ ){0,1}(do|does|does)[^\w]*( |$) to x.text
    # If the regex matches, return 1 else return ABSTAIN
    return 1 if re.search(r"( |^)(how|what|what)[^\w]* (\w+ ){0,1}(do|does|does)[^\w]*( |$)", x.text.lower()) else ABSTAIN


@labeling_function()
def description_label_55(x):
    # Apply the regex ( |^)(enumerate|list out|name|name)[^\w]* (\w+ ){0,1}(the|the)[^\w]* (\w+ ){0,1}(various|various)[^\w]*( |$) to x.text
    # If the regex matches, return 1 else return ABSTAIN
    return 1 if re.search(r"( |^)(enumerate|list out|name|name)[^\w]* (\w+ ){0,1}(the|the)[^\w]* (\w+ ){0,1}(various|various)[^\w]*( |$)", x.text.lower()) else ABSTAIN


@labeling_function()
def numeric_label_56(x):
    # Apply the regex ( |^)(at|in|in)[^\w]* (\w+ ){0,1}(which|how many|what|what)[^\w]* (\w+ ){0,1}(age|year|year)[^\w]*( |$) to x.text
    # If the regex matches, return 1 else return ABSTAIN
    return 1 if re.search(r"( |^)(at|in|in)[^\w]* (\w+ ){0,1}(which|how many|what|what)[^\w]* (\w+ ){0,1}(age|year|year)[^\w]*( |$)", x.text.lower()) else ABSTAIN


@labeling_function()
def entity_label_57(x):
    # Apply the regex ( |^)(which|what|what)[^\w]* (\w+ ){0,1}(play|game|movie|book|book)[^\w]*( |$) to x.text
    # If the regex matches, return 1 else return ABSTAIN
    return 1 if re.search(r"( |^)(which|what|what)[^\w]* (\w+ ){0,1}(play|game|movie|book|book)[^\w]*( |$)", x.text.lower()) else ABSTAIN


@labeling_function()
def human_label_58(x):
    # Apply the regex ( |^)(who|what|what)[^\w]* ([^\s]+ )*(lives|lives)[^\w]*( |$) to x.text
    # If the regex matches, return 1 else return ABSTAIN
    return 1 if re.search(r"( |^)(who|what|what)[^\w]* ([^\s]+ )*(lives|lives)[^\w]*( |$)", x.text.lower()) else ABSTAIN


@labeling_function()
def entity_label_59(x):
    # Apply the regex ( |^)(which|what|what)[^\w]* ([^\s]+ )*(organization|trust|company|company)[^\w]*( |$) to x.text
    # If the regex matches, return 1 else return ABSTAIN
    return 1 if re.search(r"( |^)(which|what|what)[^\w]* ([^\s]+ )*(organization|trust|company|company)[^\w]*( |$)", x.text.lower()) else ABSTAIN


@labeling_function()
def numeric_label_60(x):
    # Apply the regex ( |^)(latitude|latitude)[^\w]* ([^\s]+ )*(longitude|longitude)[^\w]*( |$) to x.text
    # If the regex matches, return 1 else return ABSTAIN
    return 1 if re.search(r"( |^)(latitude|latitude)[^\w]* ([^\s]+ )*(longitude|longitude)[^\w]*( |$)", x.text.lower()) else ABSTAIN


@labeling_function()
def human_label_61(x):
    # Apply the regex ( |^)(called|alias|nicknamed|nicknamed)[^\w]*( |$) to x.text
    # If the regex matches, return 1 else return ABSTAIN
    return 1 if re.search(r"( |^)(called|alias|nicknamed|nicknamed)[^\w]*( |$)", x.text.lower()) else ABSTAIN


@labeling_function()
def human_label_62(x):
    # Apply the regex ( |^)(which|who|who)[^\w]* (\w+ ){0,1}(is|will|are|was|was)[^\w]* ([^\s]+ )*(engineer|actor|actress|player|lawyer|model|captain|team|doctor|doctor)[^\w]*( |$) to x.text
    # If the regex matches, return 1 else return ABSTAIN
    return 1 if re.search(r"( |^)(which|who|who)[^\w]* (\w+ ){0,1}(is|will|are|was|was)[^\w]* ([^\s]+ )*(engineer|actor|actress|player|lawyer|model|captain|team|doctor|doctor)[^\w]*( |$)", x.text.lower()) else ABSTAIN


@labeling_function()
def location_label_63(x):
    # Apply the regex ( |^)(where|where)[^\w]*( |$) to x.text
    # If the regex matches, return 1 else return ABSTAIN
    return 1 if re.search(r"( |^)(where|where)[^\w]*( |$)", x.text.lower()) else ABSTAIN


@labeling_function()
def numeric_label_64(x):
    # Apply the regex ( |^)(by how|how|how)[^\w]* (\w+ ){0,1}(much|many|many)[^\w]*( |$) to x.text
    # If the regex matches, return 1 else return ABSTAIN
    return 1 if re.search(r"( |^)(by how|how|how)[^\w]* (\w+ ){0,1}(much|many|many)[^\w]*( |$)", x.text.lower()) else ABSTAIN


@labeling_function()
def numeric_label_65(x):
    # Apply the regex ( |^)(how|how)[^\w]* (\w+ ){0,1}(many|many)[^\w]*( |$) to x.text
    # If the regex matches, return 1 else return ABSTAIN
    return 1 if re.search(r"( |^)(how|how)[^\w]* (\w+ ){0,1}(many|many)[^\w]*( |$)", x.text.lower()) else ABSTAIN


@labeling_function()
def location_label_66(x):
    # Apply the regex ( |^)(where|where)[^\w]* (\w+ ){0,1}(was|is|is)[^\w]*( |$) to x.text
    # If the regex matches, return 1 else return ABSTAIN
    return 1 if re.search(r"( |^)(where|where)[^\w]* (\w+ ){0,1}(was|is|is)[^\w]*( |$)", x.text.lower()) else ABSTAIN


@labeling_function()
def entity_label_67(x):
    # Apply the regex ( |^)(what|what)[^\w]* (\w+ ){0,1}(is|is)[^\w]* ([^\s]+ )*(surname|address|name|name)[^\w]*( |$) to x.text
    # If the regex matches, return 1 else return ABSTAIN
    return 1 if re.search(r"( |^)(what|what)[^\w]* (\w+ ){0,1}(is|is)[^\w]* ([^\s]+ )*(surname|address|name|name)[^\w]*( |$)", x.text.lower()) else ABSTAIN


@labeling_function()
def abbreviation_label_68(x):
    # Apply the regex ( |^)(what|what)[^\w]* (\w+ ){0,1}(does|does)[^\w]* ([^\s]+ )*(stand for)[^\w]*( |$) to x.text
    # If the regex matches, return 1 else return ABSTAIN
    return 1 if re.search(r"( |^)(what|what)[^\w]* (\w+ ){0,1}(does|does)[^\w]* ([^\s]+ )*(stand for)[^\w]*( |$)", x.text.lower()) else ABSTAIN
