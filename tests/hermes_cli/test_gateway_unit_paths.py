"""User systemd unit location and identity: the account home owns the unit dir (#98699) and a bare
``hermes-gateway.service`` pinning THIS home is this home's service (#109476)."""

import pytest

from hermes_cli import gateway

pytestmark = pytest.mark.platforms("linux")


def test_user_unit_dir_follows_the_account_home_not_a_profile_pinned_process_home(tmp_path, monkeypatch):
    # ``hermes -p x gateway install`` launched from a process whose HOME profile isolation already
    # pointed at the ACTIVE profile's ``{HERMES_HOME}/home``; the unit must land where
    # ``systemctl --user`` looks — under the account home — not under that profile dir.
    account_home = tmp_path / "account"
    active_root = tmp_path / "active-profile-root"
    process_home = active_root / "home"
    process_home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(active_root))
    monkeypatch.setenv("HOME", str(process_home))
    monkeypatch.setenv("HERMES_REAL_HOME", str(account_home))
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)

    unit_path = gateway.get_systemd_unit_path(system=False)

    assert unit_path.parent == account_home / ".config" / "systemd" / "user"
    assert not unit_path.is_relative_to(process_home)


def test_bare_user_unit_pinning_this_custom_home_is_adopted_by_lifecycle_commands(tmp_path, monkeypatch):
    # A pre-#106611 install of a custom root left ``hermes-gateway.service`` (bare) pinning that root;
    # the recomputed ``hermes-gateway-<hash>`` name made it invisible and ``restart`` ran foreground.
    custom_root = tmp_path / "hermes"
    custom_root.mkdir()
    unit_dir = tmp_path / "xdg" / "systemd" / "user"
    unit_dir.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(custom_root))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.setattr(gateway, "is_linux", lambda: True)
    monkeypatch.setattr(gateway.os, "geteuid", lambda: 1000)
    monkeypatch.setattr(gateway, "supports_systemd_services", lambda: True)
    legacy_unit = unit_dir / "hermes-gateway.service"

    # A bare unit that pins ANOTHER home is someone else's service: this home keeps its own name.
    legacy_unit.write_text(
        f'[Service]\nEnvironment="HERMES_HOME={tmp_path / "other"}"\nExecStart=/usr/bin/hermes gateway run\n',
        encoding="utf-8",
    )
    assert gateway.get_service_name() != "hermes-gateway"
    assert not gateway._systemd_unit_installed()

    legacy_unit.write_text(
        f'[Service]\nEnvironment="HERMES_HOME={custom_root}"\nRestart=always\nExecStart=/usr/bin/hermes gateway run\n',
        encoding="utf-8",
    )
    assert gateway.get_service_name() == "hermes-gateway"
    assert gateway.get_systemd_unit_path(system=False) == legacy_unit
    assert gateway._systemd_unit_installed()


def test_default_named_system_unit_adopted_when_bare_unit_absent(tmp_path, monkeypatch):
    """When ``hermes-gateway.service`` is absent and ``hermes-gateway-default.service`` exists
    pinning the default home, CLI commands and sudo callers resolve ``hermes-gateway-default``."""
    import hermes_constants

    account_home = tmp_path / "home" / "tom"
    default_hermes = account_home / ".hermes"
    default_hermes.mkdir(parents=True)
    user_dir = tmp_path / "xdg" / "systemd" / "user"
    sys_dir = tmp_path / "etc" / "systemd" / "system"
    user_dir.mkdir(parents=True)
    sys_dir.mkdir(parents=True)
    monkeypatch.setattr(hermes_constants, "_account_home", lambda: account_home)
    monkeypatch.setenv("HOME", str(account_home))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.setenv("HERMES_HOME", str(default_hermes))
    monkeypatch.setattr(gateway, "is_linux", lambda: True)
    monkeypatch.setattr(gateway, "_SYSTEM_UNIT_DIR", sys_dir)
    monkeypatch.setattr(gateway.os, "geteuid", lambda: 1000)

    default_unit = sys_dir / "hermes-gateway-default.service"
    default_unit.write_text(
        f'[Service]\nEnvironment="HERMES_HOME={default_hermes}"\nExecStart=/usr/bin/hermes gateway run\n',
        encoding="utf-8",
    )

    assert gateway.get_service_name() == "hermes-gateway-default"
    assert gateway.get_systemd_unit_path(system=True) == default_unit
    assert gateway._select_systemd_scope(False) is True

    # Under sudo before unit sync, root's default home still resolves to hermes-gateway-default.
    root_hermes = tmp_path / "root" / ".hermes"
    root_hermes.mkdir(parents=True)
    monkeypatch.setattr(gateway.os, "geteuid", lambda: 0)
    monkeypatch.setattr(hermes_constants, "sudo_invoker_default_home", lambda: default_hermes)
    monkeypatch.setenv("HERMES_HOME", str(root_hermes))
    monkeypatch.setattr(hermes_constants, "_account_home", lambda: tmp_path / "root")
    assert gateway.get_service_name() == "hermes-gateway-default"

    # If bare hermes-gateway.service is present, it keeps precedence.
    bare_unit = sys_dir / "hermes-gateway.service"
    bare_unit.write_text(f'[Service]\nEnvironment="HERMES_HOME={default_hermes}"\n', encoding="utf-8")
    assert gateway.get_service_name() == "hermes-gateway"
