from unittest.mock import Mock, patch

import pytest
import yaml

from llm_ollama.cache import Cache


@pytest.fixture
def cache_dir(tmp_path):
    """Fixture providing a temporary directory for cache files."""
    return tmp_path / "cache"


@pytest.fixture
def cache(cache_dir):
    """Fixture providing a Cache instance for testing."""
    return Cache(cache_dir)


def func(value):
    """A sample function to be used in tests."""
    return f"result-{value}"


@pytest.fixture
def func_instrumented():
    """Fixture providing an instrumented version of the sample function."""
    return Mock(side_effect=func)


@pytest.fixture
def func_decorated(cache, func_instrumented):
    """Fixture providing the sample function decorated for caching."""
    return cache("sample", key="value")(func_instrumented)


class TestCacheInitialization:
    """Tests for Cache initialization."""

    def test_init_creates_directory(self, cache_dir):
        """Test that the cache creates its directory if it does not exist."""
        assert not cache_dir.exists()
        Cache(cache_dir)
        assert cache_dir.exists()
        assert cache_dir.is_dir()


class TestCacheDecorator:
    """Tests for the @cache decorator."""

    def test_basic_caching(self, cache, cache_dir, func_instrumented):
        """Test basic function caching functionality."""

        func_decorated = cache("sample", key="value")(
            lambda value: func_instrumented(value),
        )

        assert func_decorated("foo") == func("foo")
        assert func_instrumented.call_count == 1

        assert func_decorated("foo") == func("foo")
        assert func_instrumented.call_count == 1

        assert func_decorated("bar") == func("bar")
        assert func_instrumented.call_count == 2

        assert func_decorated("foo") == func("foo")
        assert func_instrumented.call_count == 2

        cache_file = cache_dir / "sample.yaml"
        assert cache_file.exists()

        with cache_file.open("r") as f:
            cache_data = yaml.safe_load(f)

        assert cache_data["version"] == Cache.CACHE_VERSION
        assert "data" in cache_data
        assert cache_data["data"]["foo"] == func("foo")
        assert cache_data["data"]["bar"] == func("bar")

    def test_arg_passing_styles(self, cache, func_instrumented):
        """Test caching with different argument passing styles."""

        @cache("sample", key="value")
        def func_with_multiple_args(value, other=None):
            return func_instrumented(value)

        # First call
        result1 = func_with_multiple_args("foobar")
        assert result1 == func("foobar")
        assert func_instrumented.call_count == 1

        result2 = func_with_multiple_args(value="foobar")
        assert result2 == func("foobar")
        assert func_instrumented.call_count == 1

    def test_key_not_provided(self, cache):
        """Test error handling when key parameter is not provided."""

        @cache("sample", key="baz")
        def function_without_baz_parameter(value, other=None):
            return func(value)

        with pytest.raises(ValueError, match="Parameter 'baz' not provided"):
            function_without_baz_parameter("test")

    def test_non_string_key_name(self, cache):
        """Test error handling for non-string key names."""
        with pytest.raises(TypeError, match="Key must be a string parameter name"):
            cache("sample", key=123)(func)

    def test_non_serializable_cache_key(self, cache):
        """Test error handling for non-serializable cache keys."""

        class NonSerializable:
            def __str__(self):
                raise TypeError("Cannot convert to string")

        @cache("sample", key="value")
        def test_func(value):
            return f"result-{value}"

        with pytest.raises(ValueError, match="not serializable for YAML"):
            test_func(NonSerializable())


class TestCacheInvalidation:
    """Tests for cache invalidation mechanisms."""

    def test_version_invalidation(self, cache_dir, func_instrumented):
        """Test that changing cache version invalidates the cache."""

        cache1 = Cache(cache_dir)
        func_decorated1 = cache1("sample", key="value")(
            lambda value: func_instrumented(value),
        )

        func_decorated1("test")
        assert func_instrumented.call_count == 1
        func_decorated1("test")
        assert func_instrumented.call_count == 1

        # Create a new cache with a different version
        current_cache_version = Cache.CACHE_VERSION
        with patch.object(Cache, "CACHE_VERSION", current_cache_version + 1):
            cache2 = Cache(cache_dir)
            func_decorated2 = cache2("sample", key="value")(
                lambda value: func_instrumented(value),
            )

            # Should not use old cache due to version change
            func_decorated2("test")
            assert func_instrumented.call_count == 2

    @pytest.mark.parametrize(
        "content",
        [
            "",  # Empty file
            "# Just a comment",  # Comment only
            "invalid: yaml: content:",  # Invalid YAML
        ],
    )
    def test_invalid_cache_file(self, cache, cache_dir, func_instrumented, content):
        """Test handling of invalid/corrupted cache files."""

        cache_file = cache_dir / "invalid.yaml"
        cache_file.write_text(content)

        func_decorated = cache("invalid", key="value")(
            lambda value: func_instrumented(value),
        )

        result = func_decorated("test")
        assert result == func("test")
        assert func_instrumented.call_count == 1

        # Check that the cache file was properly repaired/recreated
        with cache_file.open("r") as f:
            cache_data = yaml.safe_load(f)
        assert cache_data["version"] == Cache.CACHE_VERSION
        assert "test" in cache_data["data"]


class TestCacheDecoratorWithMethods:
    """Tests for caching on methods and more complex scenarios."""

    def test_class_method_caching(self, cache):
        """Test caching on class methods."""
        func_instrumented = Mock(side_effect=lambda self_val, val: f"{self_val}-{val}")

        class TestClass:
            def __init__(self, instance_value="instance"):
                self.instance_value = instance_value

            @cache("methods", key="value")
            def test_method(self, value):
                return func_instrumented(self.instance_value, value)

        # Create two instances
        instance1 = TestClass()
        instance2 = TestClass("second")

        # First call on first instance
        result1 = instance1.test_method("test")
        assert result1 == "instance-test"
        assert func_instrumented.call_count == 1

        # Call on second instance with same key - should use cache despite different
        # instance state
        result2 = instance2.test_method("test")
        assert result2 == "instance-test"  # Not "second-test"
        assert func_instrumented.call_count == 1

        # Change instance state and call again - should still use cache
        instance1.instance_value = "modified"
        result3 = instance1.test_method("test")
        assert result3 == "instance-test"  # Not "modified-test"
        assert func_instrumented.call_count == 1
