# traj_retrieval/core/webarena_handler.py
# WebArena-specific implementation of EnvironmentHandler

import os
import sys
import json
import re
import subprocess
import unicodedata
from pathlib import Path
from typing import List, Dict, Optional, Any
from dotenv import load_dotenv
from urllib.parse import urlsplit, urlunsplit


WEBARENA_SITE_ENV_KEYS = {
    "SHOPPING": 7770,
    "SHOPPING_ADMIN": 7780,
    "REDDIT": 9999,
    "GITLAB": 8023,
    "WIKIPEDIA": 8888,
    "MAP": 3000,
    "HOMEPAGE": 4399,
}

TRANSIENT_HTTP_ERROR_MARKERS = (
    "502 bad gateway",
    "503 service unavailable",
    "504 gateway time-out",
    "504 gateway timeout",
    "upstream timed out",
    "temporarily unavailable",
    "nginx/1.22.1",
)


def _normalize_webarena_host(host_raw: str) -> str:
    host = (host_raw or "").strip().strip('"').strip("'")
    if not host:
        return ""
    if host.startswith("http://") or host.startswith("https://"):
        parsed = urlsplit(host)
        host = parsed.netloc or parsed.path
    host = host.rstrip("/")
    if re.fullmatch(r"\d+(?:-\d+){3}", host):
        return f"ec2-{host}.us-east-2.compute.amazonaws.com"
    return host


def _build_webarena_site_urls(host: str, scheme: str = "http") -> Dict[str, str]:
    base = f"{scheme}://{host}"
    return {
        "SHOPPING": f"{base}:7770",
        "SHOPPING_ADMIN": f"{base}:7780/admin",
        "REDDIT": f"{base}:9999",
        "GITLAB": f"{base}:8023",
        "WIKIPEDIA": f"{base}:8888/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing",
        "MAP": f"{base}:3000",
        "HOMEPAGE": f"{base}:4399",
    }


def _current_webarena_site_urls() -> Dict[str, str]:
    return {key: os.environ.get(key, "") for key in WEBARENA_SITE_ENV_KEYS}


def _apply_webarena_host_override_from_env() -> None:
    raw_host = os.environ.get("WEBARENA_HOST", "")
    host = _normalize_webarena_host(raw_host)
    if not host:
        # Don't raise here — module import must succeed even before the user
        # has wired up WEBARENA_HOST. WebArenaHandler.__init__ revalidates and
        # raises with a clear message if the variable is still missing when an
        # actual WebArena episode is about to run.
        return
    scheme = (os.environ.get("WEBARENA_SCHEME") or "http").strip().lower() or "http"
    site_urls = _build_webarena_site_urls(host, scheme=scheme)
    os.environ["WEBARENA_HOST_RESOLVED"] = host
    for key, value in site_urls.items():
        os.environ[key] = value
    print(f"[WebArena] Applied runtime host override: {host}", flush=True)
    print(f"[WebArena] ✓ SHOPPING URL override: {site_urls['SHOPPING']}", flush=True)


def _rewrite_single_webarena_url(url: Any, site_urls: Dict[str, str]) -> Any:
    if not isinstance(url, str) or not url.startswith(("http://", "https://")):
        return url
    try:
        parsed = urlsplit(url)
        if parsed.port is None:
            return url
        target_url = next(
            (
                candidate
                for candidate in site_urls.values()
                if candidate and urlsplit(candidate).port == parsed.port
            ),
            "",
        )
        if not target_url:
            return url
        target = urlsplit(target_url)
        return urlunsplit(
            (target.scheme, target.netloc, parsed.path, parsed.query, parsed.fragment)
        )
    except Exception:
        return url


def _rewrite_webarena_url(url: Any, site_urls: Dict[str, str]) -> Any:
    if not isinstance(url, str):
        return url

    # Some WebArena tasks encode multiple startup tabs as "url1 |AND| url2".
    # Rewrite each component separately so hidden old-host URLs do not survive.
    if "|AND|" in url:
        parts = [part.strip() for part in url.split("|AND|")]
        return " |AND| ".join(
            _rewrite_single_webarena_url(part, site_urls) for part in parts
        )

    return _rewrite_single_webarena_url(url, site_urls)


def _rewrite_webarena_config_urls(value: Any, site_urls: Dict[str, str]) -> Any:
    if isinstance(value, dict):
        return {
            k: _rewrite_webarena_config_urls(v, site_urls) for k, v in value.items()
        }
    if isinstance(value, list):
        return [_rewrite_webarena_config_urls(item, site_urls) for item in value]
    return _rewrite_webarena_url(value, site_urls)


def _rewrite_prompt_intro_urls(intro: str, site_urls: Dict[str, str]) -> str:
    replacements = {
        r"(?m)^- OneStopShop \(E-commerce\): .*$": f"- OneStopShop (E-commerce): {site_urls.get('SHOPPING', '')}",
        r"(?m)^- Shopping Admin \(CMS\): .*$": f"- Shopping Admin (CMS): {site_urls.get('SHOPPING_ADMIN', '')}",
        r"(?m)^- Reddit \(Forum\): .*$": f"- Reddit (Forum): {site_urls.get('REDDIT', '')}",
        r"(?m)^- GitLab: .*$": f"- GitLab: {site_urls.get('GITLAB', '')}",
        r"(?m)^- Wikipedia: .*$": f"- Wikipedia: {site_urls.get('WIKIPEDIA', '')}",
        r"(?m)^- OpenStreetMap: .*$": f"- OpenStreetMap: {site_urls.get('MAP', '')}",
    }
    rewritten = intro
    for pattern, replacement in replacements.items():
        if replacement:
            rewritten = re.sub(pattern, replacement, rewritten)
    return rewritten


def _get_site_comb_from_cookie_filename(file_name: str) -> List[str]:
    return os.path.basename(file_name).rsplit("_", 1)[0].split(".")


def _renew_auth_cookies_subprocess(
    site_comb: List[str], auth_folder: str, timeout_s: int = 180
) -> subprocess.CompletedProcess:
    """
    Run WebArena cookie regeneration in a subprocess.

    This isolates Playwright Sync API state from the main evaluation process, so a
    failed login does not pollute later env.reset() calls with a running asyncio loop.
    """
    script = r"""
import json
import os
import sys
from pathlib import Path

WEBARENA_PATH = os.environ["WEBARENA_RENEW_WEBARENA_PATH"]
if WEBARENA_PATH not in sys.path:
    sys.path.insert(0, WEBARENA_PATH)

from playwright.sync_api import sync_playwright
from browser_env.env_config import ACCOUNTS, GITLAB, REDDIT, SHOPPING, SHOPPING_ADMIN

site_comb = json.loads(os.environ["WEBARENA_RENEW_SITES"])
auth_folder = os.environ["WEBARENA_RENEW_AUTH_FOLDER"]
Path(auth_folder).mkdir(parents=True, exist_ok=True)

with sync_playwright() as playwright:
    browser = playwright.chromium.launch(headless=True)
    context = browser.new_context()
    context.set_default_timeout(120000)
    context.set_default_navigation_timeout(120000)
    page = context.new_page()
    page.set_default_timeout(120000)
    page.set_default_navigation_timeout(120000)

    if "shopping" in site_comb:
        username = ACCOUNTS["shopping"]["username"]
        password = ACCOUNTS["shopping"]["password"]
        page.goto(f"{SHOPPING}/customer/account/login/")
        page.get_by_label("Email", exact=True).fill(username)
        page.get_by_label("Password", exact=True).fill(password)
        page.get_by_role("button", name="Sign In").click(no_wait_after=True)
        page.wait_for_timeout(3000)
        wishlist_ready = False
        for _ in range(12):
            page.goto(f"{SHOPPING}/wishlist/", wait_until="domcontentloaded")
            page.wait_for_timeout(1000)
            page_text = page.content()
            if (
                "Customer Login" not in page_text
                and "Create New Customer Account" not in page_text
                and "My Wish List" in page_text
            ):
                wishlist_ready = True
                break
        if not wishlist_ready:
            raise RuntimeError(
                f"Shopping login validation failed for {SHOPPING}; wishlist page never reached authenticated state."
            )

    if "reddit" in site_comb:
        username = ACCOUNTS["reddit"]["username"]
        password = ACCOUNTS["reddit"]["password"]
        page.goto(f"{REDDIT}/login")
        page.get_by_label("Username").fill(username)
        page.get_by_label("Password").fill(password)
        page.get_by_role("button", name="Log in").click(no_wait_after=True)
        page.wait_for_timeout(3000)

    if "shopping_admin" in site_comb:
        username = ACCOUNTS["shopping_admin"]["username"]
        password = ACCOUNTS["shopping_admin"]["password"]
        page.goto(f"{SHOPPING_ADMIN}")
        page.get_by_placeholder("user name").fill(username)
        page.get_by_placeholder("password").fill(password)
        page.get_by_role("button", name="Sign in").click(no_wait_after=True)
        page.wait_for_timeout(3000)

    if "gitlab" in site_comb:
        username = ACCOUNTS["gitlab"]["username"]
        password = ACCOUNTS["gitlab"]["password"]
        page.goto(f"{GITLAB}/users/sign_in")
        page.get_by_test_id("username-field").click()
        page.get_by_test_id("username-field").fill(username)
        page.get_by_test_id("username-field").press("Tab")
        page.get_by_test_id("password-field").fill(password)
        page.get_by_test_id("sign-in-button").click(no_wait_after=True)
        page.wait_for_timeout(3000)

    context.storage_state(path=f"{auth_folder}/{'.'.join(site_comb)}_state.json")
"""

    env = os.environ.copy()
    env["WEBARENA_RENEW_SITES"] = json.dumps(site_comb)
    env["WEBARENA_RENEW_AUTH_FOLDER"] = auth_folder
    env["WEBARENA_RENEW_WEBARENA_PATH"] = WEBARENA_PATH
    pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{WEBARENA_PATH}:{pythonpath}" if pythonpath else WEBARENA_PATH

    return subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout_s,
        check=True,
    )


# Load environment variables from .env file (for WebArena site URLs)
# This must be done BEFORE importing WebArena components
env_file = Path(__file__).parent.parent.parent / ".env"
if env_file.exists():
    load_dotenv(
        env_file, override=True
    )  # override=True ensures it replaces existing values
    _apply_webarena_host_override_from_env()
    print(f"[WebArena] Loaded environment variables from: {env_file}")
    # Verify critical variables are loaded
    import os as _os

    if _os.environ.get("SHOPPING"):
        print(f"[WebArena] ✓ SHOPPING URL: {_os.environ.get('SHOPPING')[:50]}...")
    else:
        print(f"[WebArena] ⚠️  SHOPPING URL not set!")

# Add WebArena to Python path
WEBARENA_PATH = os.path.join(os.environ.get("MATM_DATA_ROOT", "environments"), "webarena/webarena")
if os.path.exists(WEBARENA_PATH) and WEBARENA_PATH not in sys.path:
    sys.path.insert(0, WEBARENA_PATH)

# Import base handler
from .base_handler import EnvironmentHandler, TrajectoryNode, StepResult

# Import WebArena components (lazy import with error handling)
try:
    from browser_env import (
        ScriptBrowserEnv,
        Action,
        ActionTypes,
        create_id_based_action,
        create_stop_action,
        create_none_action,
        ActionParsingError,
    )
    from browser_env.utils import DetachedPage, StateInfo

    # Note: evaluator_router imported lazily in _evaluate_current_state()
    # to avoid requiring WebArena site URLs at import time
    WEBARENA_AVAILABLE = True
except ImportError as e:
    print(f"[WARNING] WebArena not available: {e}")
    WEBARENA_AVAILABLE = False
    # Define dummy types for type hints when WebArena not available
    Action = Dict  # type: ignore
    ActionTypes = None  # type: ignore
    StateInfo = Dict  # type: ignore

# WebArena uses Playwright's sync API, which requires synchronous execution
# The evaluation runner will detect requires_sync_execution=True and run this handler synchronously


class WebArenaHandler(EnvironmentHandler):
    """
    WebArena-specific implementation of EnvironmentHandler.

    WebArena is a realistic web environment with actual websites where agents
    perform complex tasks like shopping, booking, searching, and data analysis.

    Each instance creates its own independent browser environment for concurrent execution.

    Note: WebArena uses Playwright's sync API, so this handler requires sync execution.
    The async runner will automatically handle this via run_in_executor.
    """

    # Flag to indicate this environment requires synchronous execution
    # The async runner will use asyncio.run_in_executor for reset/step/close
    requires_sync_execution = True

    def __init__(
        self,
        headless: bool = True,
        slow_mo: int = 0,
        observation_type: str = "accessibility_tree",
        action_set_tag: str = "id_accessibility_tree",
        viewport_width: int = 1280,
        viewport_height: int = 720,
        current_viewport_only: bool = True,
        save_trace_enabled: bool = False,
        sleep_after_execution: float = 0.0,
        dom_settle_timeout: int = 10000,  # NEW: DOM settle timeout in milliseconds (default: 10 seconds)
    ):
        """
        Initialize WebArena handler with browser configuration.

        Args:
            headless: Run browser in headless mode
            slow_mo: Slow down browser by specified milliseconds (for debugging)
            observation_type: Type of observation ("accessibility_tree" or "html")
            action_set_tag: Action format ("id_accessibility_tree" or "playwright")
            viewport_width: Browser viewport width
            viewport_height: Browser viewport height
            current_viewport_only: Only observe current viewport (reduces obs size)
            save_trace_enabled: Enable Playwright tracing for debugging
            sleep_after_execution: Sleep duration after each action (seconds)
            dom_settle_timeout: Timeout for DOM to settle before extracting observation (milliseconds, default: 10000 = 10s)

        Raises:
            ImportError: If WebArena is not installed or configured properly
        """
        if not WEBARENA_AVAILABLE:
            raise ImportError(
                "WebArena is not installed or configured properly. "
                "Ensure WebArena is available at: " + WEBARENA_PATH
            )

        # Fail loudly here rather than later inside Playwright if the user
        # forgot to wire up the deployment hostname.
        if not _normalize_webarena_host(os.environ.get("WEBARENA_HOST", "")):
            raise RuntimeError(
                "WEBARENA_HOST is not set. WebArenaHandler needs the hostname "
                "(or host:port) of your deployed WebArena instance so it can "
                "rewrite SHOPPING / REDDIT / GITLAB / WIKIPEDIA / MAP / HOMEPAGE "
                "URLs. Set WEBARENA_HOST in your shell or .env file before "
                "starting evaluation (see .env.example)."
            )

        # Browser configuration
        self.headless = headless
        self.slow_mo = slow_mo
        self.observation_type = observation_type
        self.action_set_tag = action_set_tag
        self.viewport_size = {"width": viewport_width, "height": viewport_height}
        self.current_viewport_only = current_viewport_only
        self.save_trace_enabled = save_trace_enabled
        self.sleep_after_execution = sleep_after_execution
        self.dom_settle_timeout = dom_settle_timeout  # NEW: Store DOM settle timeout

        # Create browser environment (reuse WebArena's ScriptBrowserEnv)
        # Note: Creation happens in sync context, actual usage will be via executor in async runner
        # Pass dom_settle_timeout if WebArena supports it, otherwise it will be ignored
        try:
            self.env = ScriptBrowserEnv(
                headless=self.headless,
                slow_mo=self.slow_mo,
                observation_type=self.observation_type,
                current_viewport_only=self.current_viewport_only,
                viewport_size=self.viewport_size,
                save_trace_enabled=self.save_trace_enabled,
                sleep_after_execution=self.sleep_after_execution,
                dom_settle_timeout=self.dom_settle_timeout,  # NEW: Pass DOM settle timeout
            )
        except TypeError as e:
            # If WebArena doesn't support dom_settle_timeout parameter, fall back without it
            if "dom_settle_timeout" in str(e):
                print(
                    f"[{self.environment_name}] ⚠️  WebArena version doesn't support dom_settle_timeout parameter",
                    flush=True,
                )
                print(
                    f"[{self.environment_name}]   Falling back to default timeout (may be 500ms)",
                    flush=True,
                )
                print(
                    f"[{self.environment_name}]   Consider updating WebArena or using sleep_after_execution as workaround",
                    flush=True,
                )
                self.env = ScriptBrowserEnv(
                    headless=self.headless,
                    slow_mo=self.slow_mo,
                    observation_type=self.observation_type,
                    current_viewport_only=self.current_viewport_only,
                    viewport_size=self.viewport_size,
                    save_trace_enabled=self.save_trace_enabled,
                    sleep_after_execution=self.sleep_after_execution,
                )
            else:
                raise

        # Task-specific state
        self.config_file: Optional[str] = None
        self.task_id: int = 0
        self.task_name: str = ""  # Will be set to "intent_{intent_template_id}"
        self.intent: str = ""
        self.intent_template_id: int = 0
        self.variation_idx: str = (
            "task_id_0"  # Format: "task_id_{N}" (e.g., "task_id_756")
        )
        self.max_steps: int = 30  # Default max steps

        # State tracking
        self._last_observation: str = ""
        self._last_info: Dict = {}
        self._step_count: int = 0
        self._trajectory: List[StateInfo] = []
        self._current_obs_nodes_info: Dict = {}  # For action validation

        # Evaluator (loaded when needed)
        self.evaluator = None

    @property
    def environment_name(self) -> str:
        return "WebArena"

    def parse_evaluation_set(self, evaluation_set_data: Any) -> List[Dict[str, Any]]:
        """
        Parse WebArena evaluation set into episode descriptors.

        Expected format: Flat list of episode dicts

        Each episode MUST contain:
        - task_name: Task identifier (e.g., "intent_template_id_247") - used directly, no transformation
        - variation_id: Variation identifier (e.g., "task_id_462") - used directly, no transformation
        - config_file: Path to WebArena config file (e.g., "...config_files/0.json")

        Optional fields:
        - task_id: Numeric task ID (used internally for backward compatibility)
        - intent_template_id: Numeric intent template ID (used internally)
        - max_steps: Maximum steps (default: 30)

        Args:
            evaluation_set_data: List of episode dictionaries

        Returns:
            List of episode descriptors with task_name and variation_id preserved exactly as provided

        Raises:
            ValueError: If evaluation set format is invalid or required fields are missing
        """
        episodes = []

        # Evaluation set must be a flat list
        if not isinstance(evaluation_set_data, list):
            raise ValueError(
                f"WebArena evaluation set must be a flat list. "
                f"Got {type(evaluation_set_data).__name__}."
            )

        episode_list = evaluation_set_data
        print(
            f"[{self.environment_name}] Loaded {len(episode_list)} episodes from evaluation set"
        )

        for idx, item in enumerate(episode_list):
            # Required field: config_file path
            config_file = item.get("config_file")
            if not config_file:
                raise ValueError(
                    f"Episode {idx}: Missing required field 'config_file' in evaluation set item"
                )

            # Verify config file exists
            config_path = Path(config_file)
            if not config_path.exists():
                raise ValueError(f"Episode {idx}: Config file not found: {config_file}")

            # Load config file to extract metadata
            with open(config_path, "r") as f:
                config_data = json.load(f)

            # Use task_name and variation_id directly from evaluation set (no transformation)
            task_name = item.get("task_name")
            variation_id = item.get("variation_id")

            if not task_name:
                raise ValueError(
                    f"Episode {idx}: Missing required field 'task_name' in evaluation set item"
                )
            if not variation_id:
                raise ValueError(
                    f"Episode {idx}: Missing required field 'variation_id' in evaluation set item"
                )

            # Extract numeric values for backward compatibility (used internally)
            task_id = item.get("task_id", config_data.get("task_id", idx))
            intent_template_id = item.get(
                "intent_template_id", config_data.get("intent_template_id", 0)
            )
            intent = config_data.get("intent", "")
            max_steps = item.get("max_steps", 30)  # Default to 30 if not specified

            episode_descriptor = {
                "episode_id": f"{task_name}_{variation_id}",
                "task_name": task_name,  # Use directly from evaluation set
                "variation_id": variation_id,  # Use directly from evaluation set (e.g., "task_id_756")
                "variation_idx": variation_id,  # Use variation_id directly (matches LanceDB format: "task_id_756")
                "task_id": task_id,  # Keep numeric version for internal use only
                "intent_template_id": intent_template_id,
                "max_steps": max_steps,
                "config_file": str(config_path.absolute()),
                "intent": intent,
                "metadata": {
                    "config_data": config_data,
                    "sites": config_data.get("sites", []),
                    "start_url": config_data.get("start_url", ""),
                },
            }

            if "consumer_model" in item:
                episode_descriptor["consumer_model"] = item["consumer_model"]
            if "consumer_split_seed" in item:
                episode_descriptor["consumer_split_seed"] = item["consumer_split_seed"]
            if "source_split" in item:
                episode_descriptor["source_split"] = item["source_split"]

            # Extract simulate_step_k if present (for simulate_till_tk strategy)
            if "simulate_step_k" in item:
                episode_descriptor["simulate_step_k"] = item["simulate_step_k"]

            episodes.append(episode_descriptor)

        print(
            f"[{self.environment_name}] ✅ Parsed {len(episodes)} episodes from evaluation set",
            flush=True,
        )

        return episodes

    def initialize_from_episode(self, episode_descriptor: Dict[str, Any]) -> None:
        """Initialize from episode descriptor."""
        self.config_file = episode_descriptor["config_file"]
        self.task_id = episode_descriptor["task_id"]
        self.task_name = episode_descriptor["task_name"]
        self.intent = episode_descriptor["intent"]
        self.intent_template_id = episode_descriptor["intent_template_id"]
        self.variation_idx = episode_descriptor["variation_idx"]
        self.max_steps = episode_descriptor["max_steps"]

        # Initialize the environment
        self.initialize(
            task_name=self.task_name,
            max_steps=self.max_steps,
            variation_idx=self.variation_idx,
        )

    def initialize(
        self, task_name: str, max_steps: int, variation_idx: Optional[str] = None
    ) -> None:
        """
        Initialize WebArena environment.

        Args:
            task_name: Task name (format: "intent_template_id_{N}")
            max_steps: Maximum steps for the episode
            variation_idx: Variation identifier (format: "task_id_{N}", e.g., "task_id_756")

        Raises:
            RuntimeError: If environment initialization fails
        """
        self.task_name = task_name
        self.max_steps = max_steps
        self.variation_idx = variation_idx if variation_idx is not None else "task_id_0"

        print(
            f"[{self.environment_name}] Initializing task: {task_name}, variation: {self.variation_idx}",
            flush=True,
        )
        print(f"[{self.environment_name}] Config file: {self.config_file}", flush=True)
        print(f"[{self.environment_name}] Intent: {self.intent}", flush=True)

        # Log browser configuration
        print(f"[{self.environment_name}] Browser Config:", flush=True)
        print(f"[{self.environment_name}]   - headless: {self.headless}", flush=True)
        print(
            f"[{self.environment_name}]   - observation_type: {self.observation_type}",
            flush=True,
        )
        print(
            f"[{self.environment_name}]   - action_set_tag: {self.action_set_tag}",
            flush=True,
        )
        print(
            f"[{self.environment_name}]   - viewport: {self.viewport_size['width']}x{self.viewport_size['height']}",
            flush=True,
        )
        print(
            f"[{self.environment_name}]   - dom_settle_timeout: {self.dom_settle_timeout}ms",
            flush=True,
        )

        # Setup the environment with config file
        # Need to regenerate authentication cookies (like original WebArena run.py)
        try:
            config_path = Path(self.config_file)

            # Load config
            with open(config_path, "r") as f:
                config_data = json.load(f)

            config_data = _rewrite_webarena_config_urls(
                config_data, _current_webarena_site_urls()
            )

            # Regenerate authentication cookies if storage_state is specified
            # This matches the behavior of the original WebArena run.py
            if config_data.get("storage_state"):
                import tempfile

                storage_state_path = config_data["storage_state"]
                cookie_file_name = os.path.basename(storage_state_path)
                site_comb = _get_site_comb_from_cookie_filename(cookie_file_name)
                temp_dir = tempfile.mkdtemp()

                try:
                    print(
                        f"[{self.environment_name}] Regenerating authentication cookies for sites: {site_comb}",
                        flush=True,
                    )
                    print(
                        f"[{self.environment_name}] Updated env_config: SHOPPING={os.environ.get('SHOPPING', '')[:50]}...",
                        flush=True,
                    )
                    _renew_auth_cookies_subprocess(site_comb, temp_dir)

                    fresh_cookie_path = f"{temp_dir}/{cookie_file_name}"
                    if os.path.exists(fresh_cookie_path):
                        config_data["storage_state"] = fresh_cookie_path
                        print(
                            f"[{self.environment_name}] ✅ Generated fresh authentication cookies: {fresh_cookie_path}",
                            flush=True,
                        )
                        self._auth_temp_dir = temp_dir
                    else:
                        try:
                            temp_files = os.listdir(temp_dir)
                            print(
                                f"[{self.environment_name}] Files in temp_dir: {temp_files}",
                                flush=True,
                            )
                        except Exception:
                            pass
                        print(
                            f"[{self.environment_name}] ⚠️  Fresh cookie file not found: {fresh_cookie_path}",
                            flush=True,
                        )
                        print(
                            f"[{self.environment_name}] Expected filename: {cookie_file_name}",
                            flush=True,
                        )
                        resolved_path = (
                            config_path.parent.parent / storage_state_path
                        ).resolve()
                        if resolved_path.exists():
                            config_data["storage_state"] = str(resolved_path)
                            print(
                                f"[{self.environment_name}] Using existing auth file: {resolved_path}",
                                flush=True,
                            )
                        else:
                            config_data["storage_state"] = None
                            print(
                                f"[{self.environment_name}] Auth file not found, will continue without authentication",
                                flush=True,
                            )

                except Exception as renew_error:
                    print(
                        f"[{self.environment_name}] ⚠️  Error calling renew_comb subprocess: {renew_error}",
                        flush=True,
                    )
                    if isinstance(renew_error, subprocess.CalledProcessError):
                        if renew_error.stdout.strip():
                            print(
                                f"[{self.environment_name}] renew_comb stdout:\n{renew_error.stdout[:2000]}",
                                flush=True,
                            )
                        if renew_error.stderr.strip():
                            print(
                                f"[{self.environment_name}] renew_comb stderr:\n{renew_error.stderr[:2000]}",
                                flush=True,
                            )
                    import traceback

                    print(
                        f"[{self.environment_name}] Traceback: {traceback.format_exc()[:500]}",
                        flush=True,
                    )
                    resolved_path = (
                        config_path.parent.parent / storage_state_path
                    ).resolve()
                    if resolved_path.exists():
                        config_data["storage_state"] = str(resolved_path)
                        print(
                            f"[{self.environment_name}] Using existing auth file: {resolved_path}",
                            flush=True,
                        )
                    else:
                        config_data["storage_state"] = None
                except Exception as e:
                    print(
                        f"[{self.environment_name}] ⚠️  Error regenerating cookies: {e}",
                        flush=True,
                    )
                    config_data["storage_state"] = None

            # Save modified config to temporary file
            import tempfile

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False
            ) as tmp_config:
                json.dump(config_data, tmp_config, indent=2)
                tmp_config_path = tmp_config.name

            # Detailed asyncio debugging before Playwright setup
            print(
                f"[{self.environment_name}] ===== ASYNCIO STATE BEFORE PLAYWRIGHT =====",
                flush=True,
            )
            import asyncio

            # Check if asyncio is imported
            print(
                f"[{self.environment_name}] asyncio in sys.modules: {'asyncio' in sys.modules}",
                flush=True,
            )

            # Check for running loop
            try:
                running_loop = asyncio.get_running_loop()
                print(
                    f"[{self.environment_name}] ❌ RUNNING LOOP FOUND: {running_loop}",
                    flush=True,
                )
                print(
                    f"[{self.environment_name}]    Loop type: {type(running_loop)}",
                    flush=True,
                )
                print(
                    f"[{self.environment_name}]    Is running: {running_loop.is_running()}",
                    flush=True,
                )
                print(
                    f"[{self.environment_name}]    Is closed: {running_loop.is_closed()}",
                    flush=True,
                )
            except RuntimeError as e:
                print(f"[{self.environment_name}] ✓ No running loop: {e}", flush=True)

            # Check current event loop
            try:
                current_loop = asyncio.get_event_loop()
                print(
                    f"[{self.environment_name}] Current event loop: {current_loop}",
                    flush=True,
                )
                print(
                    f"[{self.environment_name}]    Loop type: {type(current_loop)}",
                    flush=True,
                )
                print(
                    f"[{self.environment_name}]    Is running: {current_loop.is_running()}",
                    flush=True,
                )
                print(
                    f"[{self.environment_name}]    Is closed: {current_loop.is_closed()}",
                    flush=True,
                )
            except RuntimeError as e:
                print(
                    f"[{self.environment_name}] No current event loop: {e}", flush=True
                )

            # Check event loop policy
            try:
                policy = asyncio.get_event_loop_policy()
                print(
                    f"[{self.environment_name}] Event loop policy: {policy}", flush=True
                )
                print(
                    f"[{self.environment_name}]    Policy type: {type(policy)}",
                    flush=True,
                )
            except Exception as e:
                print(
                    f"[{self.environment_name}] No event loop policy: {e}", flush=True
                )

            print(
                f"[{self.environment_name}] ==========================================",
                flush=True,
            )

            # DO NOT call setup() here!
            # WebArena's reset() method will call setup() automatically
            # Calling it twice causes the "asyncio loop running" error
            print(
                f"[{self.environment_name}] ✓ Skipping setup() - will be called by reset()",
                flush=True,
            )

            # Keep one rewritten config path for the full episode so browser
            # reset() and evaluator() see the same host-rewritten URLs.
            self._active_config_path = tmp_config_path

            print(
                f"[{self.environment_name}] ✅ Browser environment initialized (setup will happen in reset())",
                flush=True,
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to setup WebArena environment with config {self.config_file}: {e}"
            ) from e

        # Initialize evaluator for this task (lazy load to avoid import issues)
        # CRITICAL: Evaluator is required for scoring - fail fast if it can't be loaded
        try:
            from evaluation_harness import evaluator_router

            evaluator_config = getattr(self, "_active_config_path", self.config_file)
            self.evaluator = evaluator_router(evaluator_config)
            print(f"[{self.environment_name}] ✅ Evaluator loaded", flush=True)
        except Exception as e:
            import traceback

            error_msg = (
                f"[{self.environment_name}] ❌ CRITICAL: Failed to load evaluator!\n"
                f"Evaluator is required for WebArena task scoring.\n"
                f"Config file: {self.config_file}\n"
                f"Error: {str(e)}\n"
                f"Traceback:\n{traceback.format_exc()}"
            )
            print(error_msg, flush=True)
            raise RuntimeError(
                f"Failed to load WebArena evaluator for config {self.config_file}: {e}"
            ) from e

    def reset(self) -> TrajectoryNode:
        """Reset environment and return initial state."""
        if self.config_file is None:
            raise RuntimeError(
                "Config file not set. Call initialize_from_episode() first."
            )

        # Detailed asyncio debugging before reset (which calls setup() again)
        print(
            f"[{self.environment_name}] ===== ASYNCIO STATE BEFORE RESET =====",
            flush=True,
        )
        import asyncio

        try:
            running_loop = asyncio.get_running_loop()
            print(
                f"[{self.environment_name}] ❌ RUNNING LOOP FOUND: {running_loop}",
                flush=True,
            )
            print(
                f"[{self.environment_name}]    Is running: {running_loop.is_running()}",
                flush=True,
            )
        except RuntimeError as e:
            print(f"[{self.environment_name}] ✓ No running loop: {e}", flush=True)

        try:
            current_loop = asyncio.get_event_loop()
            print(
                f"[{self.environment_name}] Current event loop: {current_loop}",
                flush=True,
            )
            print(
                f"[{self.environment_name}]    Is running: {current_loop.is_running()}",
                flush=True,
            )
            print(
                f"[{self.environment_name}]    Is closed: {current_loop.is_closed()}",
                flush=True,
            )
        except RuntimeError as e:
            print(f"[{self.environment_name}] No current event loop: {e}", flush=True)
        print(
            f"[{self.environment_name}] ==========================================",
            flush=True,
        )

        # Reset WebArena environment
        # Note: reset() internally calls setup() which creates Playwright context
        # Use the rewritten per-episode config if available.
        config_to_use = getattr(self, "_active_config_path", self.config_file)
        print(
            f"[{self.environment_name}] Calling self.env.reset() with config: {config_to_use}",
            flush=True,
        )
        obs, info = self.env.reset(options={"config_file": config_to_use})
        print(
            f"[{self.environment_name}] self.env.reset() completed successfully",
            flush=True,
        )

        # CRITICAL: Set higher Playwright timeout to prevent timeout errors
        # Default timeout is 30 seconds, which is too aggressive for complex tasks
        # Increase to 120 seconds (2 minutes) to handle slow-loading pages
        try:
            if hasattr(self.env, "context") and self.env.context:
                # Set default navigation timeout on the browser context
                self.env.context.set_default_navigation_timeout(
                    120000
                )  # 120 seconds in milliseconds
                print(
                    f"[{self.environment_name}] ✓ Set Playwright navigation timeout to 120s",
                    flush=True,
                )

                # Also set timeout on all pages in the context
                for page in self.env.context.pages:
                    page.set_default_navigation_timeout(120000)  # 120 seconds
                    page.set_default_timeout(120000)  # Set general timeout too
                print(
                    f"[{self.environment_name}] ✓ Set timeout on {len(self.env.context.pages)} page(s)",
                    flush=True,
                )
            else:
                print(
                    f"[{self.environment_name}] ⚠️  Could not set timeout: context not available",
                    flush=True,
                )
        except Exception as e:
            print(
                f"[{self.environment_name}] ⚠️  Warning: Could not set Playwright timeout: {e}",
                flush=True,
            )

        # Store state
        self._last_observation = obs
        self._last_info = info
        self._step_count = 0
        self._trajectory = [{"observation": obs, "info": info}]

        # Convert observation to text
        observation_text = self._observation_to_text(obs, info)

        # Extract admissible actions from observation metadata
        admissible_actions = self._extract_admissible_actions(obs, info)

        print(f"[{self.environment_name}] Intent: '{self.intent}'", flush=True)
        print(
            f"[{self.environment_name}] Initial observation length: {len(observation_text)} chars",
            flush=True,
        )
        print(
            f"[{self.environment_name}] Initial admissible actions: {len(admissible_actions)}",
            flush=True,
        )
        if admissible_actions:
            print(
                f"[{self.environment_name}] Sample actions: {admissible_actions[:5]}",
                flush=True,
            )

        return TrajectoryNode(
            observation=observation_text,
            goal=self.intent,
            admissible_actions=admissible_actions,
            inventory="",  # WebArena doesn't have inventory
            step_number=0,
            internal_step_count=0,
            done=False,
            extra_info={
                "max_steps": self.max_steps,
                "task_id": self.task_id,
                "intent_template_id": self.intent_template_id,
                "url": self._get_current_url(info),
            },
        )

    def step(self, action: str, current_node: TrajectoryNode) -> StepResult:
        """Execute an action in WebArena."""
        if self.env is None:
            raise RuntimeError("Environment not initialized. Call initialize() first.")

        # Convert text action to WebArena action format
        webarena_action = self._text_to_webarena_action(action)

        # Check if this is a STOP action BEFORE executing
        # STOP actions don't execute any browser operation - they signal episode end
        is_stop_action = webarena_action.get("action_type") == ActionTypes.STOP

        if is_stop_action:
            # STOP action: Don't execute in environment, use current state
            # This matches WebArena's original behavior (run.py line 318-319)
            obs = self._last_observation
            info = self._last_info
            reward = 0.0  # STOP actions don't provide reward
            done = True  # STOP actions always terminate the episode
            print(
                f"[{self.environment_name}] STOP action detected: {webarena_action.get('answer', '')}",
                flush=True,
            )
        else:
            # Execute in WebArena environment
            # Note: In async context, this will be called via run_in_executor by the runner
            try:
                previous_obs = self._last_observation
                previous_info = self._last_info
                obs, reward, terminated, truncated, info = self.env.step(
                    webarena_action
                )
                # IMPORTANT: WebArena's environment doesn't set terminated=True for STOP actions
                # We handle STOP above, but for other actions, check termination flags
                done = terminated or truncated

                # Ensure any new pages created during action execution also have timeout set
                self._apply_default_timeouts()
                obs, reward, done, info = self._recover_from_transient_http_error(
                    action_text=action,
                    previous_info=previous_info,
                    obs=obs,
                    reward=reward,
                    done=done,
                    info=info,
                )
                obs, reward, done, info = self._recover_from_noop_transition(
                    action_text=action,
                    webarena_action=webarena_action,
                    previous_obs=previous_obs,
                    previous_info=previous_info,
                    obs=obs,
                    reward=reward,
                    done=done,
                    info=info,
                )

            except Exception as e:
                print(
                    f"[{self.environment_name}] Action execution failed: {e}",
                    flush=True,
                )
                import traceback

                print(
                    f"[{self.environment_name}] Traceback: {traceback.format_exc()[:500]}",
                    flush=True,
                )
                # Return current state with no reward on error
                obs = self._last_observation
                reward = 0.0
                done = False
                info = self._last_info

        # Store state
        self._last_observation = obs
        self._last_info = info
        self._step_count += 1

        # Add to trajectory (both action and resulting state)
        # Format: [StateInfo, Action, StateInfo, Action, ...]
        # This matches WebArena's trajectory format expected by evaluators
        # CRITICAL: Ensure STOP actions have "answer" field for evaluator
        # WebArena evaluator expects last_action["answer"] to exist
        # IMPORTANT: For STOP actions, the action itself should be the last item in trajectory
        # (not followed by state info), as the evaluator accesses last_action["answer"]
        if is_stop_action:
            # Ensure answer field exists (even if empty)
            if "answer" not in webarena_action:
                webarena_action["answer"] = ""
            # Debug: Log the answer for troubleshooting
            print(
                f"[{self.environment_name}] STOP action answer: '{webarena_action.get('answer', 'MISSING')}'",
                flush=True,
            )
            # For STOP actions, append only the action (no state info after)
            # This ensures the evaluator can access last_action["answer"]
            self._trajectory.append(webarena_action)
        else:
            # For non-STOP actions, append both action and resulting state
            self._trajectory.append(webarena_action)
            self._trajectory.append({"observation": obs, "info": info})

        # Convert observation to text
        observation_text = self._observation_to_text(obs, info)

        # Evaluate task completion using WebArena's evaluator
        # Evaluation happens when episode is done (STOP action or terminated/truncated)
        if done:
            print(
                f"[{self.environment_name}] Episode done - evaluating task completion...",
                flush=True,
            )
            score = self._evaluate_current_state()
            print(f"[{self.environment_name}] Evaluation score: {score}", flush=True)
        else:
            score = 0.0

        return StepResult(
            observation=observation_text,
            reward=float(reward),
            done=bool(done),
            score=float(score),
            info=info,
            internal_steps_consumed=1,
        )

    def _apply_default_timeouts(self) -> None:
        """Ensure all live Playwright pages use the longer timeout budget."""
        try:
            if hasattr(self.env, "context") and self.env.context:
                for page in self.env.context.pages:
                    try:
                        page.set_default_navigation_timeout(120000)
                        page.set_default_timeout(120000)
                    except Exception:
                        pass
        except Exception:
            pass

    def _build_info_from_live_page(self, fail_error: str = "") -> Dict[str, Any]:
        """Rebuild WebArena info from the current Playwright page."""
        page = self.env.page
        return {
            "page": DetachedPage(page.url, page.content()),
            "fail_error": fail_error,
            "observation_metadata": self.env._get_obs_metadata(),
        }

    def _refresh_current_state(
        self, fail_error: str = ""
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """Re-grab observation and metadata without consuming another agent step."""
        return self.env._get_obs(), self._build_info_from_live_page(
            fail_error=fail_error
        )

    def _is_navigation_like_action(self, action_text: str) -> bool:
        normalized = action_text.strip().lower()
        return normalized.startswith(
            (
                "click ",
                "goto ",
                "go_back",
                "go_forward",
                "press [enter]",
                "press [return]",
                "new_tab",
                "tab_focus",
            )
        )

    def _is_retryable_noop_action(self, action_text: str) -> bool:
        normalized = action_text.strip().lower()
        return normalized.startswith(
            (
                "click ",
                "goto ",
                "go_back",
                "go_forward",
                "press [enter]",
                "press [return]",
            )
        )

    def _is_transient_http_error_page(
        self, obs: Dict[str, Any], info: Dict[str, Any]
    ) -> bool:
        observation_text = self._preprocess_observation(
            self._observation_to_text(obs, info)
        ).lower()
        fail_error = ((info or {}).get("fail_error") or "").strip().lower()

        for marker in TRANSIENT_HTTP_ERROR_MARKERS:
            if marker in observation_text or marker in fail_error:
                return True
        return False

    def _recover_from_transient_http_error(
        self,
        action_text: str,
        previous_info: Dict[str, Any],
        obs: Dict[str, Any],
        reward: float,
        done: bool,
        info: Dict[str, Any],
    ) -> tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        if done or not self._is_transient_http_error_page(obs, info):
            return obs, reward, done, info

        page = getattr(self.env, "page", None)
        if page is None:
            return obs, reward, done, info

        current_url = self._get_current_url(info) or getattr(page, "url", "")
        previous_url = self._get_current_url(previous_info)
        print(
            f"[{self.environment_name}] Detected transient HTTP error page after '{action_text}' "
            f"(url={current_url}, fail_error={info.get('fail_error', '')!r})",
            flush=True,
        )

        for attempt, wait_ms in enumerate((2000, 5000), start=1):
            try:
                page.wait_for_timeout(wait_ms)
                if current_url:
                    page.goto(
                        current_url, wait_until="domcontentloaded", timeout=120000
                    )
                else:
                    page.reload(wait_until="domcontentloaded", timeout=120000)
                self._apply_default_timeouts()
                page.wait_for_timeout(1500)
                refreshed_obs, refreshed_info = self._refresh_current_state()
                if not self._is_transient_http_error_page(
                    refreshed_obs, refreshed_info
                ):
                    print(
                        f"[{self.environment_name}] Recovered from transient HTTP error after refresh attempt {attempt}",
                        flush=True,
                    )
                    return refreshed_obs, reward, done, refreshed_info
            except Exception as refresh_error:
                print(
                    f"[{self.environment_name}] Warning: transient HTTP error refresh attempt {attempt} failed: {refresh_error}",
                    flush=True,
                )

        if (
            previous_url
            and previous_url != current_url
            and self._is_navigation_like_action(action_text)
        ):
            try:
                print(
                    f"[{self.environment_name}] Falling back to browser back() after persistent transient HTTP error",
                    flush=True,
                )
                page.go_back(wait_until="domcontentloaded", timeout=120000)
                self._apply_default_timeouts()
                page.wait_for_timeout(1500)
                fallback_obs, fallback_info = self._refresh_current_state()
                if not self._is_transient_http_error_page(fallback_obs, fallback_info):
                    print(
                        f"[{self.environment_name}] Recovered from transient HTTP error by returning to previous page",
                        flush=True,
                    )
                    return fallback_obs, reward, False, fallback_info
            except Exception as back_error:
                print(
                    f"[{self.environment_name}] Warning: browser back() fallback after transient HTTP error failed: {back_error}",
                    flush=True,
                )

        return obs, reward, done, info

    def _looks_like_noop_transition(
        self,
        action_text: str,
        previous_obs: Dict[str, Any],
        previous_info: Dict[str, Any],
        obs: Dict[str, Any],
        info: Dict[str, Any],
    ) -> bool:
        if not self._is_navigation_like_action(action_text):
            return False

        fail_error = (info or {}).get("fail_error", "")
        if fail_error:
            return True

        previous_url = self._get_current_url(previous_info)
        current_url = self._get_current_url(info)
        previous_text = self._preprocess_observation(
            self._observation_to_text(previous_obs, previous_info)
        )
        current_text = self._preprocess_observation(
            self._observation_to_text(obs, info)
        )
        similarity = self.compute_observation_similarity(previous_text, current_text)
        return previous_url == current_url and similarity >= 0.995

    def _recover_from_noop_transition(
        self,
        action_text: str,
        webarena_action: Action,
        previous_obs: Dict[str, Any],
        previous_info: Dict[str, Any],
        obs: Dict[str, Any],
        reward: float,
        done: bool,
        info: Dict[str, Any],
    ) -> tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        if done or not self._looks_like_noop_transition(
            action_text, previous_obs, previous_info, obs, info
        ):
            return obs, reward, done, info

        previous_url = self._get_current_url(previous_info)
        current_url = self._get_current_url(info)
        print(
            f"[{self.environment_name}] Possible no-op transition after '{action_text}' "
            f"(prev_url={previous_url}, current_url={current_url}, fail_error={info.get('fail_error', '')!r})",
            flush=True,
        )

        try:
            if hasattr(self.env, "page") and self.env.page:
                self.env.page.wait_for_timeout(3000)
            refreshed_obs, refreshed_info = self._refresh_current_state(
                fail_error=info.get("fail_error", "")
            )
            if not self._looks_like_noop_transition(
                action_text, previous_obs, previous_info, refreshed_obs, refreshed_info
            ):
                print(
                    f"[{self.environment_name}] Recovered from no-op after waiting and refreshing observation",
                    flush=True,
                )
                return refreshed_obs, reward, done, refreshed_info
        except Exception as refresh_error:
            print(
                f"[{self.environment_name}] Warning: observation refresh after no-op suspicion failed: {refresh_error}",
                flush=True,
            )

        if not self._is_retryable_noop_action(action_text):
            return obs, reward, done, info

        print(
            f"[{self.environment_name}] Retrying action once after no-op suspicion: {action_text}",
            flush=True,
        )
        retry_obs, retry_reward, terminated, truncated, retry_info = self.env.step(
            webarena_action
        )
        retry_done = terminated or truncated
        self._apply_default_timeouts()

        try:
            if hasattr(self.env, "page") and self.env.page:
                self.env.page.wait_for_timeout(1500)
            retry_obs, retry_info = self._refresh_current_state(
                fail_error=retry_info.get("fail_error", "")
            )
        except Exception as refresh_error:
            print(
                f"[{self.environment_name}] Warning: observation refresh after retry failed: {refresh_error}",
                flush=True,
            )

        return retry_obs, float(max(reward, retry_reward)), retry_done, retry_info

    def get_current_node(self, step_number: int) -> TrajectoryNode:
        """Get current state as a TrajectoryNode."""
        if self.env is None:
            raise RuntimeError("Environment not initialized. Call initialize() first.")

        # Convert current observation to text
        observation_text = self._observation_to_text(
            self._last_observation, self._last_info
        )

        # Extract admissible actions from current state
        admissible_actions = self._extract_admissible_actions(
            self._last_observation, self._last_info
        )

        return TrajectoryNode(
            observation=observation_text,
            goal=self.intent,
            admissible_actions=admissible_actions,
            inventory="",  # WebArena doesn't have inventory
            step_number=step_number,
            internal_step_count=self._step_count,
            done=False,
            extra_info={
                "max_steps": self.max_steps,
                "task_id": self.task_id,
                "intent_template_id": self.intent_template_id,
                "url": self._get_current_url(self._last_info),
            },
        )

    def close(self) -> None:
        """Close the browser environment and clean up temporary files."""
        if self.env is not None:
            print(f"[{self.environment_name}] Closing browser environment", flush=True)
            try:
                # Note: In async context, this will be called via run_in_executor by the runner
                self.env.close()
                print(
                    f"[{self.environment_name}] ✓ Browser environment closed",
                    flush=True,
                )
            except Exception as e:
                print(
                    f"[{self.environment_name}] Warning: Error closing env: {e}",
                    flush=True,
                )
                # Try to force cleanup of Playwright resources
                try:
                    if hasattr(self.env, "context") and self.env.context:
                        self.env.context.close()
                        print(
                            f"[{self.environment_name}] ✓ Forced context close",
                            flush=True,
                        )
                    if hasattr(self.env, "browser") and self.env.browser:
                        self.env.browser.close()
                        print(
                            f"[{self.environment_name}] ✓ Forced browser close",
                            flush=True,
                        )
                    if hasattr(self.env, "playwright") and self.env.playwright:
                        self.env.playwright.stop()
                        print(
                            f"[{self.environment_name}] ✓ Forced playwright stop",
                            flush=True,
                        )
                except Exception as force_error:
                    print(
                        f"[{self.environment_name}] Warning: Force cleanup failed: {force_error}",
                        flush=True,
                    )

            # Set env to None to ensure it's not reused
            self.env = None

        # Clean up authentication temp directory if it exists
        if hasattr(self, "_auth_temp_dir"):
            import shutil

            try:
                shutil.rmtree(self._auth_temp_dir)
                print(
                    f"[{self.environment_name}] ✓ Cleaned up authentication temp directory",
                    flush=True,
                )
            except Exception as e:
                print(
                    f"[{self.environment_name}] Warning: Could not clean up auth temp dir: {e}",
                    flush=True,
                )

        if hasattr(self, "_active_config_path"):
            try:
                Path(self._active_config_path).unlink()
                print(
                    f"[{self.environment_name}] ✓ Cleaned up temporary config file",
                    flush=True,
                )
            except Exception as e:
                print(
                    f"[{self.environment_name}] Warning: Could not clean up temp config: {e}",
                    flush=True,
                )
            finally:
                delattr(self, "_active_config_path")

    def get_task_metadata(self) -> Dict[str, Any]:
        """Get task metadata for logging."""
        return {
            "task_name": self.task_name,
            "variation": self.variation_idx,
            "max_variations": 1,  # WebArena uses task_id as variation
            "goal_text": self.intent,
            "max_steps": self.max_steps,
            "is_unlimited": False,
            "environment": self.environment_name,
            "task_id": self.task_id,
            "intent_template_id": self.intent_template_id,
            "config_file": self.config_file,
        }

    def is_successful_episode(self, final_score: float) -> bool:
        """
        Determine if a WebArena episode is successful.
        WebArena evaluators return 1.0 for success, 0.0 for failure.
        """
        return final_score >= 1.0

    def build_user_message(
        self,
        goal_text: str,
        observation: str,
        inventory: str,
        admissible_actions: List[str],
        recent_history_str: str = "",
        trajectory_context: str = "",
        current_step: int = 0,
        max_steps: int = 0,
        **kwargs,
    ) -> str:
        """
        Build user message for WebArena following the consistent template format.

        Format: GOAL | CURRENT STEP | RECENT HISTORY | RETRIEVED TRAJECTORY GUIDANCE | CURRENT OBSERVATION | URL

        WebArena does NOT provide explicit admissible actions. The LLM must parse
        element IDs directly from the observation (accessibility tree).

        Args:
            goal_text: The task objective/intent
            observation: Current page observation (accessibility tree with embedded IDs)
            inventory: Not used for WebArena (kept for interface consistency)
            admissible_actions: Ignored (empty list for WebArena)
            recent_history_str: History of previous steps
            trajectory_context: Retrieved trajectory for guidance
            current_step: Current step number
            max_steps: Maximum steps allowed
            **kwargs: Additional context

        Returns:
            Formatted user message string matching standard format
        """
        message_parts = []

        # Goal
        message_parts.append(f"GOAL: {goal_text}")

        # Current step
        if current_step > 0 or max_steps > 0:
            message_parts.append(f"\nCURRENT STEP: {current_step} / {max_steps}")

        # Recent history (if present) - with clear visual separation
        if recent_history_str and recent_history_str.strip():
            message_parts.append(
                "\n--- RECENT HISTORY (Previous Steps - For Reference Only) ---"
            )
            message_parts.append(recent_history_str)
            message_parts.append("--- End of Recent History ---\n")

        # Retrieved trajectory guidance (if present) - with clear visual separation
        if trajectory_context and trajectory_context.strip():
            message_parts.append(
                "\n--- RETRIEVED TRAJECTORY GUIDANCE (Reference Examples) ---"
            )
            message_parts.append(trajectory_context)
            message_parts.append("--- End of Trajectory Guidance ---\n")

        # Current observation - with clear emphasis
        message_parts.append(
            "\n>>> CURRENT OBSERVATION (Focus on This - Current State):"
        )
        message_parts.append(observation)
        message_parts.append("<<< End of Current Observation\n")

        # URL (WebArena-specific)
        url = ""
        if hasattr(self, "_last_info"):
            try:
                page = self._last_info.get("page")
                if page and hasattr(page, "url"):
                    url = page.url
            except:
                pass

        if url:
            message_parts.append(f"CURRENT URL: {url}")

        # Response format instructions (always at the end)
        message_parts.append("\n\nRESPONSE FORMAT:")
        message_parts.append("You MUST respond with valid JSON in this exact format:")
        message_parts.append(
            '{"reasoning": "Let\'s think step by step. [your detailed reasoning]", "action": "action to be taken"}'
        )
        message_parts.append("\nWhere:")
        message_parts.append(
            "- reasoning: MUST start with 'Let's think step by step.' Then explain your thought process, what you observe, and why this action is best"
        )
        message_parts.append("- action: action to be taken")
        message_parts.append("\nIMPORTANT:")
        message_parts.append(
            "1. Your reasoning MUST begin with 'Let's think step by step.'"
        )
        message_parts.append(
            "2. Do not include any text before or after the JSON object."
        )

        return "\n".join(message_parts)

    def build_system_message(self, admissible_actions: List[str]) -> str:
        """
        Build the system message for LLM prompts using WebArena prompt template.

        This loads from traj_retrieval/core/webarena_data.json with our custom format
        that emphasizes "Let's think step-by-step" reasoning followed by action in ``` delimiters.

        Returns:
            System message string that explains the task and available actions

        Raises:
            FileNotFoundError: If webarena_data.json is not found
            ValueError: If prompt file is invalid or missing required fields
        """
        # Load from our custom WebArena prompt file
        webarena_prompt_file = Path(__file__).parent / "webarena_data.json"

        if not webarena_prompt_file.exists():
            raise FileNotFoundError(
                f"[{self.environment_name}] ❌ WebArena prompt file not found: {webarena_prompt_file}\n"
                f"This file is required for correct WebArena prompt formatting."
            )

        try:
            with open(webarena_prompt_file, "r") as f:
                prompt_data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"[{self.environment_name}] ❌ Invalid JSON in WebArena prompt file: {webarena_prompt_file}\n"
                f"Error: {e}"
            ) from e

        # Validate required fields
        required_fields = ["intro", "examples", "template", "meta_data"]
        missing_fields = [f for f in required_fields if f not in prompt_data]
        if missing_fields:
            raise ValueError(
                f"[{self.environment_name}] ❌ WebArena prompt file missing required fields: {missing_fields}\n"
                f"Required: {required_fields}\n"
                f"Available: {list(prompt_data.keys())}"
            )

        meta_data = prompt_data["meta_data"]

        # Build complete system message with examples
        intro = _rewrite_prompt_intro_urls(
            prompt_data["intro"], _current_webarena_site_urls()
        )
        if not getattr(self, "_logged_prompt_site_urls", False):
            website_lines = []
            capture = False
            for line in intro.splitlines():
                if line.strip() == "Available Websites:":
                    capture = True
                    website_lines.append(line)
                    continue
                if capture:
                    if not line.strip():
                        break
                    website_lines.append(line)
            if website_lines:
                print(
                    f"[{self.environment_name}] Prompt website section after runtime override:\n"
                    + "\n".join(website_lines),
                    flush=True,
                )
            self._logged_prompt_site_urls = True
        examples = prompt_data["examples"]

        # Format examples consistently
        example_texts = []
        for i, example in enumerate(examples, 1):
            if isinstance(example, (list, tuple)) and len(example) == 2:
                user_example, assistant_example = example
                example_texts.append(
                    f"Example {i}:\n{user_example}\n\nResponse: {assistant_example}"
                )
            else:
                print(
                    f"[{self.environment_name}] ⚠️ Skipping malformed example {i}",
                    flush=True,
                )

        # Combine intro and examples
        system_message = intro
        if example_texts:
            system_message += "\n\n" + "\n\n".join(example_texts)
            system_message += "\n\nNow, given the current observation, think step-by-step and provide your response."

        print(
            f"[{self.environment_name}] ✅ Loaded WebArena prompt from: {webarena_prompt_file}",
            flush=True,
        )
        print(
            f"[{self.environment_name}]    Examples included: {len(examples)}",
            flush=True,
        )

        return system_message

    # ============================================================================
    # Private helper methods
    # ============================================================================

    def _observation_to_text(self, obs: Dict, info: Dict) -> str:
        """
        Convert WebArena's observation to text format.

        WebArena provides observations in a dict with 'text' and 'image' keys.
        For text-based agents, we use the 'text' field which contains either
        the accessibility tree or HTML content.

        Args:
            obs: Observation dict from WebArena
            info: Info dict with additional metadata

        Returns:
            Text representation of the observation
        """
        if isinstance(obs, dict) and "text" in obs:
            return obs["text"]
        elif isinstance(obs, str):
            return obs
        else:
            # Fallback: return URL if we can't get observation
            return f"Current URL: {self._get_current_url(info)}"

    def _get_current_url(self, info: Dict) -> str:
        """Extract current URL from info dict."""
        page = info.get("page")
        if page and hasattr(page, "url"):
            return page.url
        return ""

    def get_current_url(self) -> str:
        """
        Get the current URL for WebArena.

        Returns:
            Current page URL string
        """
        return self._get_current_url(self._last_info)

    def format_history_entry(
        self,
        step: int,
        observation: str,
        action: str,
        reward: float,
        reasoning: str = "",
        inventory: str = "",
        url: str = "",
    ) -> str:
        """
        Format a single history entry for WebArena.
        Includes URL but no inventory.
        """
        # Get current URL if not provided
        if not url:
            url = self.get_current_url()

        entry = f"STEP {step}: OBSERVATION: {observation}"

        # Add URL (WebArena-specific)
        if url and url.strip():
            entry += f" | URL: {url}"

        # Add action and reward
        entry += f" | ACTION: {action} | REWARD: {reward}"

        # Add reasoning if available
        if reasoning and reasoning.strip():
            entry += f" | REASONING: {reasoning}"

        return entry

    def _extract_admissible_actions(self, obs: Dict, info: Dict) -> List[str]:
        """
        WebArena does NOT use explicit admissible actions lists.
        Following the original implementation, the LLM parses element IDs directly from the observation text.

        Returns empty list to maintain interface compatibility.
        """
        # Store obs_nodes_info for validation purposes
        obs_metadata = info.get("observation_metadata", {})
        text_metadata = obs_metadata.get("text", {})
        self._current_obs_nodes_info = text_metadata.get("obs_nodes_info", {})

        # Return empty list - no explicit action enumeration for WebArena
        return []

    def _text_to_webarena_action(self, action_text: str) -> Action:
        """
        Convert text-based action to WebArena's Action format.

        Uses WebArena's action parsing functions to ensure compatibility.

        Action formats:
        - "click [123]" -> click element 123
        - "type [456] [hello world] [1]" -> type text in element 456 with Enter
        - "hover [789]" -> hover over element 789
        - "scroll [up]" or "scroll [down]" -> scroll page
        - "press [Enter]" -> press key
        - "goto [url]" -> navigate to URL (URL must be in brackets)
        - "stop []" -> stop episode
        - "none" -> no-op action (valid fallback)

        Args:
            action_text: Text-based action string

        Returns:
            WebArena Action dict
        """
        # Handle "none" action explicitly (common fallback for WebArena)
        action_text_stripped = action_text.strip().lower()
        if action_text_stripped == "none":
            return create_none_action()

        # Normalize goto actions that might be missing brackets
        # WebArena expects: "goto [url]" but LLM might generate "goto http://..."
        action_text_normalized = action_text.strip()
        if (
            action_text_normalized.lower().startswith("goto ")
            and "[" not in action_text_normalized
        ):
            # Extract URL after "goto "
            url_part = action_text_normalized[5:].strip()  # Skip "goto "
            action_text_normalized = f"goto [{url_part}]"
            print(
                f"[{self.environment_name}] Normalized goto action: '{action_text}' -> '{action_text_normalized}'",
                flush=True,
            )

        # UNICODE NORMALIZATION: Handle any Unicode characters from LLM
        # WebArena only supports ASCII (32-127) and limited Unicode (129-999)
        # Many characters outside this range (like U+2011 non-breaking hyphen at 8209) will cause errors
        original_normalized = action_text_normalized

        # Step 1: Apply Unicode normalization (NFKD = compatibility decomposition)
        # This converts composed characters to their decomposed equivalents
        action_text_normalized = unicodedata.normalize("NFKD", action_text_normalized)

        # Step 2: Replace common problematic Unicode characters with ASCII equivalents
        # These are commonly generated by LLMs but not supported by WebArena
        unicode_replacements = {
            # Various types of hyphens and dashes
            "\u2010": "-",  # hyphen
            "\u2011": "-",  # non-breaking hyphen (U+2011, decimal 8209) - THE CULPRIT
            "\u2012": "-",  # figure dash
            "\u2013": "-",  # en dash
            "\u2014": "-",  # em dash
            "\u2015": "-",  # horizontal bar
            "\u2212": "-",  # minus sign
            # Various types of quotes
            "\u2018": "'",  # left single quotation mark
            "\u2019": "'",  # right single quotation mark
            "\u201a": "'",  # single low-9 quotation mark
            "\u201b": "'",  # single high-reversed-9 quotation mark
            "\u201c": '"',  # left double quotation mark
            "\u201d": '"',  # right double quotation mark
            "\u201e": '"',  # double low-9 quotation mark
            "\u201f": '"',  # double high-reversed-9 quotation mark
            "\u2039": "<",  # single left-pointing angle quotation mark
            "\u203a": ">",  # single right-pointing angle quotation mark
            # Other common Unicode punctuation
            "\u2026": "...",  # horizontal ellipsis
            "\u2022": "*",  # bullet
            "\u2032": "'",  # prime
            "\u2033": '"',  # double prime
            "\u00a0": " ",  # non-breaking space
            "\u00ad": "",  # soft hyphen (remove it)
            "\u200b": "",  # zero-width space (remove it)
            "\u200c": "",  # zero-width non-joiner (remove it)
            "\u200d": "",  # zero-width joiner (remove it)
            "\ufeff": "",  # zero-width no-break space (BOM, remove it)
        }

        for unicode_char, ascii_replacement in unicode_replacements.items():
            action_text_normalized = action_text_normalized.replace(
                unicode_char, ascii_replacement
            )

        # Step 3: Remove any remaining non-ASCII characters that can't be converted
        # This is a safety net for any other Unicode characters
        # We use 'ignore' to skip characters that can't be encoded to ASCII
        try:
            action_text_normalized = action_text_normalized.encode(
                "ascii", "ignore"
            ).decode("ascii")
        except Exception as e:
            print(
                f"[{self.environment_name}] Warning: ASCII encoding failed: {e}",
                flush=True,
            )

        # Log if normalization made changes
        if action_text_normalized != original_normalized:
            print(
                f"[{self.environment_name}] Unicode normalization applied:", flush=True
            )
            print(
                f"[{self.environment_name}]   Before: {repr(original_normalized)}",
                flush=True,
            )
            print(
                f"[{self.environment_name}]   After:  {repr(action_text_normalized)}",
                flush=True,
            )

        try:
            # Use WebArena's built-in action parser
            if self.action_set_tag == "id_accessibility_tree":
                return create_id_based_action(action_text_normalized)
            else:
                raise ValueError(f"Unsupported action_set_tag: {self.action_set_tag}")
        except ActionParsingError as e:
            # If parsing fails, use "none" action as safe fallback instead of stop
            print(f"[{self.environment_name}] Action parsing failed: {e}", flush=True)
            print(
                f"[{self.environment_name}] Original action: '{action_text}', Normalized: '{action_text_normalized}'",
                flush=True,
            )
            print(
                f"[{self.environment_name}] Using 'none' action as safe fallback for invalid action",
                flush=True,
            )
            return create_none_action()
        except Exception as e:
            # For unexpected errors, also use "none" as safe fallback
            print(
                f"[{self.environment_name}] Unexpected error in action parsing: {e}",
                flush=True,
            )
            print(
                f"[{self.environment_name}] Original action: '{action_text}', Normalized: '{action_text_normalized}'",
                flush=True,
            )
            print(
                f"[{self.environment_name}] Using 'none' action as safe fallback",
                flush=True,
            )
            return create_none_action()

    def _evaluate_current_state(self) -> float:
        """
        Use WebArena's evaluator to check task completion.

        Returns:
            Score (1.0 for success, 0.0 for failure)

        Raises:
            RuntimeError: If evaluator is not loaded or evaluation fails
        """
        if self.evaluator is None:
            raise RuntimeError(
                f"[{self.environment_name}] Evaluator not loaded! "
                f"This should never happen - evaluator must be loaded in initialize()."
            )

        try:
            # Get page and client from environment
            page = self.env.page
            client = self.env.get_page_client(page)

            print(
                f"[{self.environment_name}] Running evaluator with trajectory length: {len(self._trajectory)}",
                flush=True,
            )
            print(f"[{self.environment_name}] Current page URL: {page.url}", flush=True)

            # Debug: Check last action in trajectory (evaluator expects last_action["answer"])
            if len(self._trajectory) > 0:
                last_item = self._trajectory[-1]
                second_last_item = (
                    self._trajectory[-2] if len(self._trajectory) > 1 else None
                )
                print(
                    f"[{self.environment_name}] Trajectory debug - Last item type: {type(last_item)}, keys: {list(last_item.keys()) if isinstance(last_item, dict) else 'not a dict'}",
                    flush=True,
                )
                if second_last_item:
                    print(
                        f"[{self.environment_name}] Trajectory debug - Second last item type: {type(second_last_item)}, keys: {list(second_last_item.keys()) if isinstance(second_last_item, dict) else 'not a dict'}",
                        flush=True,
                    )
                    # Check if second_last is a STOP action
                    if (
                        isinstance(second_last_item, dict)
                        and second_last_item.get("action_type") == ActionTypes.STOP
                    ):
                        print(
                            f"[{self.environment_name}] Trajectory debug - Second last is STOP action with answer: '{second_last_item.get('answer', 'MISSING')}'",
                            flush=True,
                        )

            # Use WebArena's evaluator with trajectory
            # Trajectory format: [StateInfo, Action, StateInfo, Action, ...]
            # Evaluator expects trajectory to end with an Action (typically STOP)
            score = self.evaluator(
                trajectory=self._trajectory,
                config_file=getattr(self, "_active_config_path", self.config_file),
                page=page,
                client=client,
            )

            print(
                f"[{self.environment_name}] Evaluator returned score: {score}",
                flush=True,
            )
            return float(score)
        except Exception as e:
            import traceback

            error_msg = (
                f"[{self.environment_name}] ❌ Evaluation failed!\n"
                f"Config file: {getattr(self, '_active_config_path', self.config_file)}\n"
                f"Trajectory length: {len(self._trajectory)}\n"
                f"Error: {str(e)}\n"
                f"Traceback:\n{traceback.format_exc()}"
            )
            print(error_msg, flush=True)
            # Don't silently fail - raise the error so we can debug
            raise RuntimeError(
                f"WebArena evaluation failed for config {getattr(self, '_active_config_path', self.config_file)}: {e}"
            ) from e

    # ============================================================================
    # Required interface methods for retrieval and prompting
    # ============================================================================

    def format_retrieval_result(
        self,
        raw_data: Dict[str, Any],
        max_steps: int = 20,
        retrieval_type: str = "trajectory",
    ) -> str:
        """Format retrieved trajectory for WebArena."""
        if retrieval_type != "trajectory":
            print(
                f"[{self.environment_name}] Unknown retrieval type: {retrieval_type}, skipping formatting"
            )
            return ""

        if not raw_data:
            return ""

        # Extract trajectory information
        trajectory_steps = raw_data.get("trajectory_steps") or raw_data.get(
            "remaining_action_observation_pairs", []
        )
        task_description = raw_data.get("task_description", "")

        if not trajectory_steps:
            return ""

        print(
            f"[{self.environment_name}] Formatting trajectory: {len(trajectory_steps)} total steps, showing {min(len(trajectory_steps), max_steps)}"
        )

        # Format the trajectory steps
        formatted_steps = []
        for i, step in enumerate(trajectory_steps[:max_steps]):
            action = step.get("action", "")
            observation = step.get("observation", "")

            # Truncate long observations for readability
            if len(observation) > 300:
                observation = observation[:300] + "..."

            formatted_steps.append(
                f"Step {i+1}:\n" f"  Action: {action}\n" f"  Observation: {observation}"
            )

        # Create the context string
        context = f"""RETRIEVED TRAJECTORY:
Task: {task_description}

Retrieved successful trajectory:
{chr(10).join(formatted_steps)}

Use this trajectory as a reference for your planning. Consider:
1. The sequence of actions taken
2. How the agent navigated through the website
3. What elements were clicked and in what order
4. When the task was completed

"""
        return context

    def get_environment_description(self) -> str:
        """Get WebArena-specific environment description."""
        return """You are interacting with real websites through a web browser.
You can see the page content as an accessibility tree showing interactive elements.
Each element has a unique ID that you use for actions."""

    def get_action_types_description(self) -> str:
        """Get WebArena-specific action types."""
        return """Available actions:
- click [id]: Click on element with given id
- type [id] [text] [1]: Type text into element (1 means press Enter after)
- hover [id]: Hover over element
- scroll [up/down]: Scroll page up or down
- press [key]: Press a keyboard key (e.g., Enter, Backspace)
- stop []: Stop and end the episode"""

    def build_retrieval_query(
        self,
        goal: str,
        observation: str,
        inventory: str = "",
        recent_history: str = "",
        current_step: int = 0,
        current_reward: float = 0.0,
    ) -> str:
        """
        Build query matching WebArena indexing format.

        Format for WebArena:
        - state: observation ONLY (no inventory)
        - progress: step_till_now + current_reward

        Args:
            goal: Task intent
            observation: Current page observation
            inventory: Not used in WebArena
            recent_history: Recent action history
            current_step: Current step number
            current_reward: Current reward/score

        Returns:
            Query string formatted to match database entries
        """
        # WebArena has no inventory, but tracks current page state
        state = f"observation: {observation}"
        context = recent_history if recent_history else ""
        progress = f"step_till_now: {current_step} | current_reward: {current_reward}"

        # Full key format (must match what's embedded in database)
        full_query = (
            f"goal: {goal} | state: {state} | context: {context} | progress: {progress}"
        )
        return full_query

    def validate_action(self, action: str, admissible_actions: List[str]) -> bool:
        """
        Validate if an action is valid for WebArena.

        WebArena uses ID-based actions like "click [123]", where the IDs come from
        the accessibility tree. We validate by checking if the element ID exists in
        the current observation's metadata.

        Args:
            action: The action to validate (e.g., "click [164]", "scroll [up]", "stop []")
            admissible_actions: Ignored for WebArena (empty list)

        Returns:
            True if action is valid, False otherwise
        """
        action = action.strip()

        # Extract action type
        action_type = (
            action.split("[")[0].strip()
            if "[" in action
            else action.split()[0]
            if " " in action
            else action
        )

        # Standard actions that don't require element IDs
        standard_actions = [
            "scroll",
            "press",
            "goto",
            "new_tab",
            "close_tab",
            "go_back",
            "go_forward",
            "tab_focus",
            "page_focus",
            "stop",
            "none",
        ]

        if action_type in standard_actions:
            # These actions are always valid regardless of current page state
            return True

        # For ID-based actions (click, type, hover), validate element ID exists
        if action_type in ["click", "type", "hover"]:
            # Extract element ID from action
            match = re.search(r"\[(\w+)\]", action)
            if match:
                element_id = match.group(1)
                # Check if this element ID exists in current observation
                if element_id in self._current_obs_nodes_info:
                    return True
                else:
                    print(
                        f"[WebArena] Invalid element ID '{element_id}' not found in current observation",
                        flush=True,
                    )
                    return False
            else:
                print(
                    f"[WebArena] Could not extract element ID from action: {action}",
                    flush=True,
                )
                return False

        # Unknown action type
        print(f"[WebArena] Unknown action type: {action_type}", flush=True)
        return False

    def _preprocess_observation(self, obs: str) -> str:
        """
        Preprocess WebArena observation by removing element ID numbers.

        Element IDs like [1], [970], [1140] change between runs, so we remove them
        to focus on the structure and content of the accessibility tree.

        Example:
            "[970] gridcell 'Grace Nguyen'" -> "gridcell 'Grace Nguyen'"
            "[1] RootWebArea 'Dashboard'" -> "RootWebArea 'Dashboard'"

        Args:
            obs: Raw observation string with element IDs

        Returns:
            Observation with [number] patterns removed
        """
        # Remove [number] patterns using regex
        # Pattern: [ followed by one or more digits, followed by ]
        import re

        processed = re.sub(r"\[\d+\]", "", obs)
        return processed

    def compute_observation_similarity(self, obs1: str, obs2: str) -> float:
        """
        Compute similarity between two WebArena observations using edit distance.

        IMPORTANT: This method expects PREPROCESSED observations (with element IDs already removed).
        Call _preprocess_observation() before passing observations to this method.

        Process:
        1. Calculate Levenshtein distance (edit distance) between preprocessed observations
        2. Convert to similarity percentage

        Args:
            obs1: First PREPROCESSED observation (accessibility tree with IDs removed)
            obs2: Second PREPROCESSED observation (accessibility tree with IDs removed)

        Returns:
            Similarity score: 1.0 for identical, 0.0 for completely different
        """
        # Calculate edit distance (Levenshtein distance)
        # Using Python's difflib for simplicity (could use python-Levenshtein for speed)
        import difflib

        # SequenceMatcher.ratio() returns similarity in range [0.0, 1.0]
        # where 1.0 = identical, 0.0 = completely different
        matcher = difflib.SequenceMatcher(None, obs1, obs2)
        similarity = matcher.ratio()

        return similarity

    def _extract_url_path(self, url: str) -> str:
        """
        Extract the path part of a URL (after domain).

        Example:
            "http://your-webarena-host:7780/admin/admin/dashboard/"
            -> "/admin/admin/dashboard/"

        Args:
            url: Full URL string

        Returns:
            Path part of URL (including leading /), or empty string if no path
        """
        if not url:
            return ""

        try:
            from urllib.parse import urlparse

            parsed = urlparse(url)
            # Return path (includes leading /)
            return parsed.path
        except Exception as e:
            print(f"[WebArena] Warning: Could not parse URL '{url}': {e}", flush=True)
            return ""

    def should_trigger_retrieval(
        self,
        current_observation: str,
        target_observation: str,
        similarity_threshold: float = 0.95,  # 95% similarity threshold
        current_url: str = "",
        target_url: str = "",
    ) -> bool:
        """
        Determine if retrieval should be triggered for WebArena based on state similarity.

        Matching logic:
        1. Preprocess observations (remove element IDs like [123])
        2. Calculate edit distance similarity
        3. If similarity >= 95%, MATCH
        4. Otherwise, check FULL URL (if target URL is not empty)

        URL matching compares the COMPLETE URL including path, query params, and fragment.
        Example: "http://example.com/projects/new#tab1" only matches "http://example.com/projects/new#tab1"

        Args:
            current_observation: Current observation from environment
            target_observation: Target observation from step k in simulation data
            similarity_threshold: Minimum similarity to trigger retrieval (default: 0.95 = 95%)
            current_url: Current URL (empty string for non-web environments)
            target_url: Target URL from step k (empty string means don't use URL for matching)

        Returns:
            True if retrieval should be triggered (observation OR URL matches), False otherwise
        """
        # Preprocess observations (remove element IDs)
        processed_current = self._preprocess_observation(current_observation)
        processed_target = self._preprocess_observation(target_observation)

        # Calculate similarity using edit distance on PREPROCESSED observations
        obs_similarity = self.compute_observation_similarity(
            processed_current, processed_target
        )
        obs_matches = obs_similarity >= similarity_threshold

        # Log preprocessing and similarity details
        print(f"\n[WebArena] ===== OBSERVATION SIMILARITY CHECK =====", flush=True)
        print(f"[WebArena] Preprocessing:", flush=True)
        print(
            f"[WebArena]   Current obs length: {len(current_observation)} chars",
            flush=True,
        )
        print(
            f"[WebArena]   Target obs length:  {len(target_observation)} chars",
            flush=True,
        )
        print(
            f"[WebArena]   Processed current length: {len(processed_current)} chars",
            flush=True,
        )
        print(
            f"[WebArena]   Processed target length:  {len(processed_target)} chars",
            flush=True,
        )

        # Show preview of processed observations (first 200 chars)
        print(
            f"[WebArena]   Processed current preview: {processed_current[:200]}...",
            flush=True,
        )
        print(
            f"[WebArena]   Processed target preview:  {processed_target[:200]}...",
            flush=True,
        )

        print(f"[WebArena] Similarity calculation:", flush=True)
        print(
            f"[WebArena]   Edit distance similarity: {obs_similarity:.4f} ({obs_similarity*100:.2f}%)",
            flush=True,
        )
        print(
            f"[WebArena]   Threshold: {similarity_threshold:.4f} ({similarity_threshold*100:.2f}%)",
            flush=True,
        )
        print(
            f"[WebArena]   Observation match: {'✓ YES' if obs_matches else '✗ NO'}",
            flush=True,
        )

        # ALWAYS check and log URL matching (for debugging)
        url_matches = False
        if target_url:  # Only check URL if target URL is provided
            # Compare FULL URLs (not just path)
            url_matches = current_url == target_url

            print(f"[WebArena] URL matching:", flush=True)
            print(f"[WebArena]   Current URL: {current_url}", flush=True)
            print(f"[WebArena]   Target URL:  {target_url}", flush=True)
            print(
                f"[WebArena]   Full URL match: {'✓ YES' if url_matches else '✗ NO'}",
                flush=True,
            )

            # If URLs don't match, show why (for debugging)
            if not url_matches:
                current_path = self._extract_url_path(current_url)
                target_path = self._extract_url_path(target_url)
                path_matches = current_path == target_path
                print(
                    f"[WebArena]   Debug - Path only: {current_path} vs {target_path} ({'match' if path_matches else 'no match'})",
                    flush=True,
                )
        else:
            print(f"[WebArena] URL matching: SKIPPED (target URL is empty)", flush=True)

        # Final result: Observation match OR URL match
        # Note: URL only contributes to match if observation didn't match
        final_match = obs_matches or (not obs_matches and url_matches)

        print(
            f"[WebArena] Final result: {'✓✓✓ MATCH (retrieval triggered)' if final_match else '✗✗✗ NO MATCH'}",
            flush=True,
        )
        if final_match:
            if obs_matches:
                match_reason = "observation similarity >= 95%"
            else:
                match_reason = "full URL match"
            print(f"[WebArena] Match reason: {match_reason}", flush=True)
        print(f"[WebArena] =========================================\n", flush=True)

        return final_match
