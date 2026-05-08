from __future__ import annotations


def search_stac_items(*_: object, **__: object) -> list[dict]:
    """STAC 검색은 첫 버전에서 최소 인터페이스만 제공한다."""
    # STAC 연동은 pystac-client 선택 의존성을 통해 이후 확장한다.
    raise NotImplementedError("STAC search support is planned for a future release.")

