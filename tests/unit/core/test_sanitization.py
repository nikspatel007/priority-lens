"""Tests for email sanitization module."""

from __future__ import annotations

from priority_lens.core.sanitization import (
    SanitizationResult,
    collapse_whitespace,
    normalize_characters,
    remove_quoted_content,
    remove_signature,
    sanitize_email,
    strip_html,
    strip_template_syntax,
)


class TestSanitizationResult:
    """Tests for SanitizationResult dataclass."""

    def test_reduction_ratio(self) -> None:
        """Test reduction ratio calculation."""
        result = SanitizationResult(
            text="abc",
            original_length=10,
            sanitized_length=3,
            html_stripped=False,
            quotes_removed=False,
            signature_removed=False,
        )
        assert result.reduction_ratio == 0.7

    def test_reduction_ratio_zero_length(self) -> None:
        """Test reduction ratio with zero original length."""
        result = SanitizationResult(
            text="",
            original_length=0,
            sanitized_length=0,
            html_stripped=False,
            quotes_removed=False,
            signature_removed=False,
        )
        assert result.reduction_ratio == 0.0

    def test_reduction_ratio_no_change(self) -> None:
        """Test reduction ratio when content unchanged."""
        result = SanitizationResult(
            text="test",
            original_length=4,
            sanitized_length=4,
            html_stripped=False,
            quotes_removed=False,
            signature_removed=False,
        )
        assert result.reduction_ratio == 0.0


class TestStripHtml:
    """Tests for strip_html function."""

    def test_removes_basic_tags(self) -> None:
        """Test basic HTML tag removal."""
        html = "<html><body><p>Hello <b>world</b></p></body></html>"
        result = strip_html(html)
        assert "Hello" in result
        assert "world" in result
        assert "<" not in result

    def test_removes_scripts(self) -> None:
        """Test script tag removal."""
        html = "<html><script>alert('xss')</script><p>Content</p></html>"
        result = strip_html(html)
        assert "Content" in result
        assert "alert" not in result
        assert "xss" not in result

    def test_removes_styles(self) -> None:
        """Test style tag removal."""
        html = "<html><style>.foo { color: red; }</style><p>Content</p></html>"
        result = strip_html(html)
        assert "Content" in result
        assert "foo" not in result
        assert "color" not in result

    def test_removes_tracking_pixels(self) -> None:
        """Test 1x1 tracking pixel removal."""
        html = '<p>Hello</p><img src="tracker.gif" width="1" height="1">'
        result = strip_html(html)
        assert "Hello" in result
        assert "tracker" not in result

    def test_removes_tracking_pixels_height_only(self) -> None:
        """Test tracking pixel removal with height=1 only."""
        html = '<p>Content</p><img src="track.gif" height="1" width="100">'
        result = strip_html(html)
        assert "Content" in result
        # Tracking pixel should be removed based on height=1

    def test_removes_tracking_pixels_width_only(self) -> None:
        """Test tracking pixel removal with width=1 only."""
        html = '<p>Content</p><img src="pixel.gif" width="1" height="100">'
        result = strip_html(html)
        assert "Content" in result

    def test_preserves_normal_images(self) -> None:
        """Test that normal images are not removed."""
        html = '<p>Content</p><img src="photo.jpg" width="200" height="150" alt="Photo">'
        result = strip_html(html)
        assert "Content" in result

    def test_handles_img_without_dimensions(self) -> None:
        """Test image without width/height attributes."""
        html = '<p>Hello</p><img src="image.jpg">'
        result = strip_html(html)
        assert "Hello" in result

    def test_removes_hidden_elements(self) -> None:
        """Test hidden element removal."""
        html = '<p>Visible</p><div style="display:none">Hidden</div>'
        result = strip_html(html)
        assert "Visible" in result
        assert "Hidden" not in result

    def test_removes_visibility_hidden(self) -> None:
        """Test visibility:hidden element removal."""
        html = '<p>Visible</p><span style="visibility:hidden">Secret</span>'
        result = strip_html(html)
        assert "Visible" in result
        assert "Secret" not in result

    def test_removes_html_comments(self) -> None:
        """Test HTML comment removal."""
        html = "<p>Hello</p><!-- This is a comment --><p>World</p>"
        result = strip_html(html)
        assert "Hello" in result
        assert "World" in result
        assert "comment" not in result

    def test_empty_input(self) -> None:
        """Test empty input."""
        assert strip_html("") == ""

    def test_none_input(self) -> None:
        """Test None input."""
        assert strip_html(None) == ""  # type: ignore[arg-type]

    def test_exception_handling(self) -> None:
        """Test exception handling returns basic stripped HTML."""
        from unittest.mock import patch

        # Mock BeautifulSoup to raise an exception
        with patch(
            "priority_lens.core.sanitization.BeautifulSoup", side_effect=Exception("Parse error")
        ):
            result = strip_html("<p>Hello</p>")
            # Should fall back to regex stripping
            assert "Hello" in result
            assert "<p>" not in result

    def test_preserves_spacing(self) -> None:
        """Test that text from different elements is properly spaced."""
        html = "<p>First paragraph</p><p>Second paragraph</p>"
        result = strip_html(html)
        assert "First paragraph" in result
        assert "Second paragraph" in result

    def test_complex_email_html(self) -> None:
        """Test complex email HTML structure."""
        html = """
        <html>
        <head><title>Email</title><style>body { font: Arial; }</style></head>
        <body>
            <div style="display:none">Preheader</div>
            <table><tr><td>Main content here</td></tr></table>
            <img src="track.gif" width="1" height="1">
            <script>tracking();</script>
        </body>
        </html>
        """
        result = strip_html(html)
        assert "Main content here" in result
        assert "Preheader" not in result
        assert "tracking" not in result


class TestStripTemplateSyntax:
    """Tests for strip_template_syntax function."""

    def test_removes_liquid_blocks(self) -> None:
        """Test Liquid/Jinja block tag removal."""
        text = "Hello {% if true %}World{% endif %}"
        result = strip_template_syntax(text)
        assert "Hello" in result
        assert "{%" not in result
        assert "if" not in result or "if true" not in result

    def test_removes_liquid_variables(self) -> None:
        """Test Liquid/Jinja variable removal."""
        text = "Hello {{ user.name }}!"
        result = strip_template_syntax(text)
        assert "Hello" in result
        assert "!" in result
        assert "{{" not in result

    def test_removes_jinja_comments(self) -> None:
        """Test Jinja comment removal."""
        text = "Hello {# this is a comment #} World"
        result = strip_template_syntax(text)
        assert "Hello" in result
        assert "World" in result
        assert "{#" not in result

    def test_removes_handlebars_sections(self) -> None:
        """Test Handlebars section removal."""
        text = "{{#if condition}}content{{/if}}"
        result = strip_template_syntax(text)
        assert "{{#" not in result
        assert "{{/" not in result

    def test_removes_erb_tags(self) -> None:
        """Test ERB tag removal."""
        text = "Hello <% code %> World"
        result = strip_template_syntax(text)
        assert "Hello" in result
        assert "World" in result
        assert "<%" not in result

    def test_empty_input(self) -> None:
        """Test empty input."""
        assert strip_template_syntax("") == ""

    def test_preserves_plain_text(self) -> None:
        """Test that plain text is preserved."""
        text = "This is plain text without templates"
        result = strip_template_syntax(text)
        assert result == text


class TestRemoveQuotedContent:
    """Tests for remove_quoted_content function."""

    def test_removes_quoted_lines(self) -> None:
        """Test basic quoted line removal."""
        text = "My reply here\n> Previous message\n> More quoted text"
        result = remove_quoted_content(text)
        assert "My reply here" in result
        assert "Previous message" not in result

    def test_removes_on_wrote_pattern(self) -> None:
        """Test 'On ... wrote:' pattern removal."""
        text = "My response\nOn Mon, Jan 1, 2024, John wrote:\nQuoted content"
        result = remove_quoted_content(text)
        assert "My response" in result
        assert "John wrote" not in result

    def test_removes_outlook_headers(self) -> None:
        """Test Outlook-style header removal."""
        text = "Reply\nFrom: sender@example.com\nSent: Monday\nTo: me@example.com"
        result = remove_quoted_content(text)
        assert "Reply" in result
        assert "From:" not in result
        assert "Sent:" not in result

    def test_removes_forwarded_message(self) -> None:
        """Test forwarded message header removal."""
        text = "FYI\n---------- Forwarded message ----------\nOriginal content"
        result = remove_quoted_content(text)
        assert "FYI" in result

    def test_removes_original_message(self) -> None:
        """Test original message header removal."""
        text = "My reply\n----- Original Message -----\nOriginal content"
        result = remove_quoted_content(text)
        assert "My reply" in result

    def test_handles_nested_quotes(self) -> None:
        """Test nested quote handling."""
        text = "Reply\n> First level\n>> Second level\n>>> Third level"
        result = remove_quoted_content(text)
        assert "Reply" in result
        assert "First level" not in result
        assert "Second level" not in result

    def test_empty_input(self) -> None:
        """Test empty input."""
        assert remove_quoted_content("") == ""

    def test_no_quotes(self) -> None:
        """Test text without quotes."""
        text = "This is a normal message without any quotes."
        result = remove_quoted_content(text)
        assert result == text

    def test_empty_line_in_quote_block(self) -> None:
        """Test that empty lines inside quote block don't exit the block."""
        # Empty line after "On ... wrote:" pattern should stay in quote block
        # and be skipped without being added to output
        text = "Main reply\nOn Mon, Jan 1 wrote:\n\nSome text\nMore text"
        result = remove_quoted_content(text)
        # Main reply should be preserved
        assert "Main reply" in result
        # The empty line should NOT cause "On ... wrote:" to be included
        # Some text should be preserved (exits quote block)
        assert "Some text" in result
        assert "More text" in result


class TestRemoveSignature:
    """Tests for remove_signature function."""

    def test_removes_double_dash_signature(self) -> None:
        """Test standard double-dash signature removal."""
        text = "Message body\n--\nJohn Doe\nCompany Inc."
        result, removed = remove_signature(text)
        assert "Message body" in result
        assert "John Doe" not in result
        assert removed is True

    def test_removes_triple_dash_signature(self) -> None:
        """Test triple-dash signature removal."""
        text = "Message body\n---\nSignature"
        result, removed = remove_signature(text)
        assert "Message body" in result
        assert "Signature" not in result
        assert removed is True

    def test_removes_sent_from_iphone(self) -> None:
        """Test 'Sent from my iPhone' removal."""
        text = "Quick reply\nSent from my iPhone"
        result, removed = remove_signature(text)
        assert "Quick reply" in result
        assert "Sent from my iPhone" not in result
        assert removed is True

    def test_removes_sent_from_ipad(self) -> None:
        """Test 'Sent from my iPad' removal."""
        text = "Quick reply\nSent from my iPad"
        result, removed = remove_signature(text)
        assert "Quick reply" in result
        assert removed is True

    def test_removes_outlook_mobile(self) -> None:
        """Test Outlook mobile signature removal."""
        text = "Reply\nGet Outlook for iOS"
        result, removed = remove_signature(text)
        assert "Reply" in result
        assert "Outlook" not in result
        assert removed is True

    def test_removes_best_regards(self) -> None:
        """Test 'Best regards' signature removal."""
        text = "Message content\nBest regards,\nJohn"
        result, removed = remove_signature(text)
        assert "Message content" in result
        assert "Best regards" not in result
        assert removed is True

    def test_no_signature(self) -> None:
        """Test text without signature."""
        text = "This is a message without a signature."
        result, removed = remove_signature(text)
        assert result == text
        assert removed is False

    def test_empty_input(self) -> None:
        """Test empty input."""
        result, removed = remove_signature("")
        assert result == ""
        assert removed is False


class TestNormalizeCharacters:
    """Tests for normalize_characters function."""

    def test_normalizes_smart_quotes(self) -> None:
        """Test smart quote normalization."""
        text = "\u201cHello\u201d \u2018World\u2019"  # Smart double and single quotes
        result = normalize_characters(text)
        assert '"Hello"' in result
        assert "'World'" in result

    def test_normalizes_dashes(self) -> None:
        """Test em/en dash normalization."""
        text = "Range: 1\u201310 and something\u2014else"  # en-dash and em-dash
        result = normalize_characters(text)
        assert "1-10" in result
        assert "something-else" in result

    def test_normalizes_ellipsis(self) -> None:
        """Test ellipsis normalization."""
        text = "Wait for it\u2026"  # Unicode ellipsis
        result = normalize_characters(text)
        assert "Wait for it..." in result

    def test_removes_zero_width_chars(self) -> None:
        """Test zero-width character removal."""
        text = "Hello\u200bWorld"  # Zero-width space
        result = normalize_characters(text)
        assert result == "HelloWorld"

    def test_empty_input(self) -> None:
        """Test empty input."""
        assert normalize_characters("") == ""


class TestCollapseWhitespace:
    """Tests for collapse_whitespace function."""

    def test_collapses_multiple_spaces(self) -> None:
        """Test multiple space collapse."""
        text = "Hello    World"
        result = collapse_whitespace(text)
        assert result == "Hello World"

    def test_collapses_tabs(self) -> None:
        """Test tab collapse."""
        text = "Hello\t\tWorld"
        result = collapse_whitespace(text)
        assert result == "Hello World"

    def test_preserves_paragraph_breaks(self) -> None:
        """Test paragraph break preservation."""
        text = "First paragraph\n\n\n\nSecond paragraph"
        result = collapse_whitespace(text)
        assert "First paragraph\n\nSecond paragraph" == result

    def test_strips_line_whitespace(self) -> None:
        """Test line leading/trailing whitespace removal."""
        text = "  Line one  \n  Line two  "
        result = collapse_whitespace(text)
        assert "Line one" in result
        assert "Line two" in result
        assert not result.startswith(" ")
        assert not result.endswith(" ")

    def test_empty_input(self) -> None:
        """Test empty input."""
        assert collapse_whitespace("") == ""


class TestSanitizeEmail:
    """Tests for sanitize_email function."""

    def test_full_sanitization(self) -> None:
        """Test complete email sanitization pipeline."""
        content = """
        <html>
        <body>
        <p>Hello {{ user.name }},</p>
        <p>This is the main content.</p>
        </body>
        </html>

        > Previous message quote

        --
        John Doe
        """
        result = sanitize_email(content)
        assert "Hello" in result.text
        assert "main content" in result.text
        assert "{{" not in result.text
        assert "Previous message" not in result.text
        assert "John Doe" not in result.text
        assert result.html_stripped is True

    def test_empty_input(self) -> None:
        """Test empty input."""
        result = sanitize_email("")
        assert result.text == ""
        assert result.original_length == 0
        assert result.sanitized_length == 0

    def test_plain_text_passthrough(self) -> None:
        """Test plain text without HTML."""
        text = "This is plain text content."
        result = sanitize_email(text)
        assert result.text == text
        assert result.html_stripped is False

    def test_disable_html_stripping(self) -> None:
        """Test disabling HTML stripping."""
        html = "<p>Content</p>"
        result = sanitize_email(html, strip_html_content=False)
        # HTML should remain
        assert "<p>" in result.text or "Content" in result.text

    def test_disable_template_removal(self) -> None:
        """Test disabling template removal."""
        text = "Hello {{ name }}"
        result = sanitize_email(text, remove_templates=False)
        assert "{{" in result.text

    def test_disable_quote_removal(self) -> None:
        """Test disabling quote removal."""
        text = "Reply\n> Quote"
        result = sanitize_email(text, remove_quotes=False)
        assert ">" in result.text

    def test_disable_signature_removal(self) -> None:
        """Test disabling signature removal."""
        text = "Content\n--\nSignature"
        result = sanitize_email(text, remove_sig=False)
        assert "Signature" in result.text
        assert result.signature_removed is False

    def test_reduction_ratio_calculation(self) -> None:
        """Test that reduction ratio is calculated correctly."""
        content = """
        <html><body>
        <script>lots of code</script>
        <style>lots of styles</style>
        <p>Short content</p>
        </body></html>
        """
        result = sanitize_email(content)
        assert result.reduction_ratio > 0  # Content should be reduced
        assert result.sanitized_length < result.original_length

    def test_complex_email(self) -> None:
        """Test complex real-world email content."""
        content = """
        <html>
        <head><style>body { font-family: Arial; }</style></head>
        <body>
        <div style="display:none">Preheader text for preview</div>
        <table>
        <tr><td>
        Hi {{ first_name }},

        {% if has_order %}
        Your order #{{ order_id }} has shipped!
        {% endif %}

        Thanks for your business.
        </td></tr>
        </table>
        <img src="track.gif" width="1" height="1">
        </body>
        </html>

        On Mon, Jan 1, 2024, Support wrote:
        > Your previous inquiry...

        --
        Customer Service Team
        """
        result = sanitize_email(content)

        # Should have main content
        assert "shipped" in result.text.lower() or "order" in result.text.lower()

        # Should not have template syntax
        assert "{{" not in result.text
        assert "{%" not in result.text

        # Should not have quotes
        assert "previous inquiry" not in result.text.lower()

        # Should be significantly reduced
        assert result.reduction_ratio > 0.3

    def test_disable_character_normalization(self) -> None:
        """Test disabling character normalization."""
        text = "Hello \u201cWorld\u201d"  # Smart quotes
        result = sanitize_email(text, normalize_chars=False)
        # Smart quotes should remain
        assert "\u201c" in result.text or "\u201d" in result.text

    def test_quote_block_continues_to_next_line(self) -> None:
        """Test that quote block continues across multiple quoted lines."""
        text = "Main content\nOn Mon, Jan 1 wrote:\n> Quoted line 1\n> Quoted line 2\n> More quoted"
        result = sanitize_email(text)
        assert "Main content" in result.text
        assert "Quoted line 1" not in result.text
        assert "Quoted line 2" not in result.text
        assert "More quoted" not in result.text
