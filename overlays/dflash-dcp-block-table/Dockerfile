ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG BASE_IMAGE_ID
ARG RECIPE_SHA256
LABEL org.nous.glm53.base-image-id="${BASE_IMAGE_ID}" \
      org.nous.glm53.overlay="dflash-dcp-block-table" \
      org.nous.glm53.overlay-recipe-sha256="${RECIPE_SHA256}" \
      org.nous.glm53.patch-origin="independently-derived" \
      org.nous.glm53.target-source-license="Apache-2.0"

COPY patch-model-runner.py /opt/glm53-overlay/patch-model-runner.py
RUN python3 /opt/glm53-overlay/patch-model-runner.py --apply \
      /usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu/model_runner.py \
 && python3 /opt/glm53-overlay/patch-model-runner.py --verify \
      /usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu/model_runner.py
