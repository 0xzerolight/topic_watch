"""Tests for the notification module: Apprise wrapper and formatting."""

from unittest.mock import MagicMock, patch

from app.analysis.llm import NoveltyResult
from app.config import LLMSettings, NotificationSettings, Settings
from app.notifications import format_notification, send_notification


def _make_settings(**overrides) -> Settings:
    defaults = {
        "llm": LLMSettings(model="openai/gpt-4o-mini", api_key="test-key"),
        "notifications": NotificationSettings(urls=["json://localhost"]),
    }
    defaults.update(overrides)
    return Settings(**defaults)


# --- format_notification ---


class TestFormatNotification:
    """Tests for notification formatting."""

    def test_title_includes_topic_name(self) -> None:
        novelty = NoveltyResult(
            has_new_info=True,
            summary="New release date announced",
            confidence=0.9,
        )
        title, body = format_notification("Elden Ring DLC", novelty)
        assert title == "Topic Watch: Elden Ring DLC"

    def test_body_includes_summary(self) -> None:
        novelty = NoveltyResult(
            has_new_info=True,
            summary="Price announced at $39.99",
            confidence=0.9,
        )
        _, body = format_notification("Test", novelty)
        assert "Price announced at $39.99" in body

    def test_body_includes_key_facts(self) -> None:
        novelty = NoveltyResult(
            has_new_info=True,
            summary="Update",
            key_facts=["Fact one", "Fact two"],
            confidence=0.9,
        )
        _, body = format_notification("Test", novelty)
        assert "Fact one" in body
        assert "Fact two" in body
        assert "Key facts:" in body

    def test_body_includes_source_urls(self) -> None:
        novelty = NoveltyResult(
            has_new_info=True,
            summary="Update",
            source_urls=["https://example.com/article"],
            confidence=0.9,
        )
        _, body = format_notification("Test", novelty)
        assert "https://example.com/article" in body
        assert "Sources:" in body

    def test_handles_no_summary(self) -> None:
        novelty = NoveltyResult(
            has_new_info=True,
            key_facts=["A fact"],
            confidence=0.9,
        )
        title, body = format_notification("Test", novelty)
        assert title == "Topic Watch: Test"
        assert "A fact" in body

    def test_handles_empty_novelty(self) -> None:
        novelty = NoveltyResult(has_new_info=True, confidence=0.8)
        title, body = format_notification("Test", novelty)
        assert title == "Topic Watch: Test"
        assert isinstance(body, str)

    def test_body_includes_confidence_percentage(self) -> None:
        novelty = NoveltyResult(
            has_new_info=True,
            summary="Update",
            confidence=0.9,
        )
        _, body = format_notification("Test", novelty)
        assert "Confidence: 90%" in body

    def test_body_confidence_truncates_to_int(self) -> None:
        novelty = NoveltyResult(
            has_new_info=True,
            summary="Update",
            confidence=0.856,
        )
        _, body = format_notification("Test", novelty)
        assert "Confidence: 85%" in body


# --- send_notification ---


class TestSendNotification:
    """Tests for the Apprise send wrapper (async)."""

    @patch("app.notifications.apprise.Apprise")
    async def test_sends_successfully(self, mock_apprise: MagicMock) -> None:
        mock_instance = MagicMock()
        mock_instance.notify.return_value = True
        mock_apprise.return_value = mock_instance
        settings = _make_settings()

        result = await send_notification("Title", "Body", settings)

        assert result is True
        mock_instance.add.assert_called_once_with("json://localhost")
        mock_instance.notify.assert_called_once_with(title="Title", body="Body")

    @patch("app.notifications.apprise.Apprise")
    async def test_adds_multiple_urls(self, mock_apprise: MagicMock) -> None:
        mock_instance = MagicMock()
        mock_instance.notify.return_value = True
        mock_apprise.return_value = mock_instance
        settings = _make_settings(
            notifications=NotificationSettings(urls=["json://localhost", "slack://token/channel"])
        )

        await send_notification("T", "B", settings)

        assert mock_instance.add.call_count == 2

    async def test_returns_false_when_no_urls(self) -> None:
        settings = _make_settings(notifications=NotificationSettings(urls=[]))
        result = await send_notification("Title", "Body", settings)
        assert result is False

    @patch("app.notifications.apprise.Apprise")
    async def test_returns_false_on_delivery_failure(self, mock_apprise: MagicMock) -> None:
        mock_instance = MagicMock()
        mock_instance.notify.return_value = False
        mock_apprise.return_value = mock_instance
        settings = _make_settings()

        result = await send_notification("Title", "Body", settings)

        assert result is False

    @patch("app.notifications.apprise.Apprise")
    async def test_returns_false_on_exception(self, mock_apprise: MagicMock) -> None:
        mock_instance = MagicMock()
        mock_instance.notify.side_effect = Exception("Connection refused")
        mock_apprise.return_value = mock_instance
        settings = _make_settings()

        result = await send_notification("Title", "Body", settings)

        assert result is False
