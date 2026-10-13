"""Model lookups on the Hugging Face Hub."""

from collections.abc import Iterator

import httpx2
from huggingface_hub import auth_check, list_models, model_info
from huggingface_hub.errors import GatedRepoError, HfHubHTTPError, RepositoryNotFoundError
from huggingface_hub.hf_api import ExpandModelProperty_T

from ..domain import ModelCheck, ModelSummary, weights_size
from ..errors import ModelError

SEARCH_FIELDS: list[ExpandModelProperty_T] = ["downloads", "likes", "gated", "safetensors"]


class HubClient:
    def check_model(self, model_id: str, token: str | None) -> ModelCheck:
        """Look up a model's weights, or raise ModelError if it can't be served."""
        try:
            info = model_info(model_id, token=token or None, files_metadata=True)
        except GatedRepoError:  # a subclass of RepositoryNotFoundError, so it must come first
            raise ModelError(f"{model_id} is gated and this token doesn't have access") from None
        except RepositoryNotFoundError:
            raise ModelError(f"{model_id} was not found on Hugging Face (or it is private and needs a token)") from None
        except HfHubHTTPError as e:
            raise ModelError(f"Couldn't look up {model_id}: {e}") from e

        weights = [s for s in info.siblings or [] if s.rfilename.endswith(".safetensors")]
        if not weights:
            raise ModelError(f"{model_id} has no .safetensors weights, which the S3 streamer requires")
        return ModelCheck(model_id=model_id, size_bytes=sum(s.size or 0 for s in weights), gated=bool(info.gated))

    def has_access(self, model_id: str, token: str) -> bool:
        try:
            auth_check(model_id, token=token)
            return True
        except (GatedRepoError, RepositoryNotFoundError):
            return False

    def search_models(
        self, query: str | None = None, trending: bool = False, max_scan: int = 200
    ) -> Iterator[ModelSummary]:
        """Text-generation models, most popular first, that have .safetensors weights."""
        listing = list_models(
            pipeline_tag="text-generation",
            search=query or None,
            sort="trending_score" if trending else "downloads",
            limit=max_scan,
            expand=SEARCH_FIELDS,
        )
        try:
            for info in listing:
                # GGUF-only and other repos without safetensors can't be streamed from S3
                if not info.safetensors or not info.safetensors.parameters:
                    continue
                yield ModelSummary(
                    model_id=info.id,
                    downloads=info.downloads or 0,
                    likes=info.likes or 0,
                    gated=bool(info.gated),
                    size_bytes=weights_size(info.safetensors.parameters),
                )
        except httpx2.HTTPError as e:  # HfHubHTTPError included
            raise ModelError(f"Couldn't reach Hugging Face: {e}") from e
