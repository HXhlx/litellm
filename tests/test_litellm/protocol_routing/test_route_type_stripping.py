"""Regression tests: proxy-internal `_route_type` must never reach providers.

`route_request()` injects ``data["_route_type"]`` so the router's protocol
filter can inspect the inbound protocol. That marker is internal: every path
that forwards to litellm/provider functions must strip it first.

Covers the reported bug where model-id testing (``has_model_id`` /
``specific_deployment=True`` early-return dict path) skipped the pop in
``async_get_healthy_deployments`` and leaked ``_route_type`` into
``input_kwargs`` for the provider call.
"""

from unittest.mock import AsyncMock, patch

import pytest

from litellm import Router


@pytest.fixture
def single_router():
    """Router with one deployment carrying a known model_info id."""
    return Router(
        model_list=[
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": "openai/gpt-4o",
                    "custom_llm_provider": "openai",
                    "api_key": "sk-fake",
                },
                "model_info": {"id": "test-deployment-id"},
            },
        ]
    )


class TestRouteTypeStrippedBeforeProvider:
    @pytest.mark.asyncio
    async def test_model_id_path_strips_route_type(self, single_router):
        """has_model_id early-return dict path must pop _route_type."""
        request_kwargs = {"_route_type": "acompletion", "metadata": {}}
        deployment = await single_router.async_get_healthy_deployments(
            model="test-deployment-id",
            request_kwargs=request_kwargs,
        )
        assert isinstance(deployment, dict)
        assert "_route_type" not in request_kwargs

    @pytest.mark.asyncio
    async def test_specific_deployment_path_strips_route_type(self, single_router):
        """specific_deployment=True path must pop _route_type."""
        request_kwargs = {"_route_type": "acompletion", "metadata": {}}
        await single_router.async_get_healthy_deployments(
            model="openai/gpt-4o",
            request_kwargs=request_kwargs,
            messages=[{"role": "user", "content": "hi"}],
            specific_deployment=True,
        )
        assert "_route_type" not in request_kwargs

    def test_sync_path_strips_route_type(self, single_router):
        """Sync get_available_deployment must pop _route_type."""
        request_kwargs = {"_route_type": "acompletion", "metadata": {}}
        deployment = single_router.get_available_deployment(
            model="test-deployment-id",
            request_kwargs=request_kwargs,
        )
        assert isinstance(deployment, dict)
        assert "_route_type" not in request_kwargs

    @pytest.mark.asyncio
    async def test_proxy_skips_injection_for_unmapped_route_type(self, single_router):
        """Route types unknown to the protocol filter get no marker at all."""
        from litellm.proxy.route_llm_request import route_request

        data = {"model": "whatever"}
        try:
            await route_request(
                data=data,
                llm_router=None,
                user_model=None,
                route_type="acreate_container",
            )
        except Exception:
            pass
        assert "_route_type" not in data

    @pytest.mark.asyncio
    async def test_proxy_strips_before_direct_litellm_call(self):
        """Direct litellm dispatches (no router) must not forward _route_type."""
        from litellm.proxy.route_llm_request import route_request

        data = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
        }
        with patch("litellm.acompletion", new=AsyncMock()) as mock_acompletion:
            result = await route_request(
                data=data,
                llm_router=None,
                user_model="test-user-model",
                route_type="acompletion",
            )
            # route_request returns the unawaited provider coroutine here.
            if hasattr(result, "__await__"):
                await result
        assert mock_acompletion.called
        assert "_route_type" not in mock_acompletion.call_args.kwargs
