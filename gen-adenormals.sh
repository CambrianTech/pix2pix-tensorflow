
NORMALS_MODEL=$checkpoints/normals512_515k
ADE_MODEL=$checkpoints/ade512
RESOLUTION=512
SRC_EXP="*.jpg"
GT_EXP="*_seg.png"
MIX_OPACITY=0.5
MIX_FILTER=subtract

NORMALS_DEST=/tmp/normals_${RESOLUTION}
ADE_DEST=/tmp/ade_${RESOLUTION}
GT_DEST=/tmp/GT_${RESOLUTION}
MIXED_DEST=/tmp/ade_normals_mix_${RESOLUTION}
OUTPUT=$datasets/ade_normals_mix_${RESOLUTION}_ab

rm -Rf ${NORMALS_DEST}
rm -Rf ${ADE_DEST}
rm -Rf ${GT_DEST}
rm -Rf ${MIXED_DEST}
rm -Rf ${OUTPUT}

python looper.py  \
--input_dir $datasets/ADE20K_2016_07_26/images/training \
--input_match_exp ${SRC_EXP} \
--output_dir ${NORMALS_DEST} \
--checkpoint ${NORMALS_MODEL} \
--filter_categories $datasets/ADE20K_2016_07_26/indoor-categories.txt \
--crop_size ${RESOLUTION}

python looper.py  \
--input_dir $datasets/ADE20K_2016_07_26/images/training \
--input_match_exp ${SRC_EXP} \
--output_dir ${ADE_DEST} \
--checkpoint ${ADE_MODEL} \
--filter_categories $datasets/ADE20K_2016_07_26/indoor-categories.txt \
--crop_size ${RESOLUTION}

python ab_combiner.py  \
--a_input_dir ${ADE_DEST}  \
--b_input_dir ${NORMALS_DEST} \
--output_dir ${MIXED_DEST} \
--filter ${MIX_FILTER} --opacity ${MIX_OPACITY}

python looper.py  \
--input_dir $datasets/ADE20K_2016_07_26/images/training \
--input_match_exp ${GT_EXP} \
--output_dir ${GT_DEST} \
--checkpoint ${ADE_MODEL} \
--filter_categories $datasets/ADE20K_2016_07_26/indoor-categories.txt \
--run_nnet=0 \
--crop_size ${RESOLUTION}

python ab_converter.py  \
--a_input_dir ${MIXED_DEST} \
--b_input_dir ${GT_DEST} \
--output_dir ${OUTPUT} \
--filter_categories $datasets/ADE20K_2016_07_26/indoor-categories.txt \
--replace_colors $datasets/ADE20K_2016_07_26/replace-colors.txt 