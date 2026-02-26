def test_load_etf_data_count(etf_data):
    assert len(etf_data) == 8


def test_etf_data_has_required_fields(etf_data):
    required_fields = [
        "id", "name", "ticker", "category", "index",
        "asset_manager", "total_expense_ratio", "risk_level"
    ]
    for etf in etf_data:
        for field in required_fields:
            assert field in etf, f"ETF {etf.get('id', '?')} missing field: {field}"


def test_create_documents_count(documents):
    assert len(documents) == 8


def test_documents_have_metadata(documents):
    for doc in documents:
        assert "id" in doc.metadata
        assert "name" in doc.metadata
        assert "ticker" in doc.metadata


def test_documents_have_content(documents):
    for doc in documents:
        assert len(doc.page_content) > 100
        assert "ETF ID:" in doc.page_content
