"""Tokamak-layer merge: device defaults inherited, scenario deltas override.

A scenario env sets ``tokamak: iter`` and inherits the packaged ITER tokamak
config (geometry plumbing, ion mix, actuators, observations, disruption); it
overlays only its scenario-specific deltas.
"""

from plasmax.environment.merge import _merge_env_and_backend
from plasmax.environment.registry import resolve_backend, resolve_env


def _merge(env, backend="cgm"):
    return _merge_env_and_backend(resolve_env(env), resolve_backend(backend))


class TokamakMergeTest:
    def test_actuators_and_disruption_inherited_from_tokamak(self):
        # Actuators/disruption live only in tokamaks/iter.yaml now; every ITER
        # scenario inherits the unified values.
        for env in (
            "iter/hybrid/flattop",
            "iter/baseline/rampup",
            "iter/advanced/flattop",
        ):
            m = _merge(env)
            acts = {a["name"]: a for a in m["actuators"]}
            assert acts["P_nbi"]["high"] == 50.0e6
            assert acts["rho_eccd"]["low"] == 0.0
            assert m["disruption"]["greenwald_metric"] == "volume_avg"
            assert m["disruption"]["greenwald_threshold"] == 1.0

    def test_scenario_deltas_override_and_survive(self):
        m = _merge("iter/baseline/flattop")
        # Env-set scalars survive the merge.
        assert m["torax"]["plasma_composition"]["Z_eff"] == 1.7
        assert m["torax"]["numerics"]["t_final"] == 200.0
        assert (
            m["torax"]["geometry"]["geometry_file"]
            == "references/iter_baseline_450s.eqdsk"
        )
        # Device-level torax fields come from the tokamak.
        assert m["torax"]["plasma_composition"]["main_ion"] == {"D": 0.5, "T": 0.5}
        assert m["torax"]["numerics"]["resistivity_multiplier"] == 1
        assert m["torax"]["geometry"]["cocos"] == 7
        # The tokamak directive itself must not leak into the merged config.
        assert "tokamak" not in m

    def test_scenario_base_shared_across_phases_with_phase_overrides(self):
        # base.yaml holds the physics common to a scenario's phases; each phase
        # file overlays only its deltas (Ip schedule, t_final, geometry, ...).
        rampup = _merge("iter/hybrid/rampup")
        flattop = _merge("iter/hybrid/flattop")
        # The physical hot reference keeps the exact upstream pedestal; the
        # constrained cold anchor keeps the upstream ramp-up pedestal.
        assert rampup["torax"]["pedestal"]["T_i_ped"] == 1.0
        assert flattop["torax"]["pedestal"]["T_i_ped"] == 4.5
        for merged in (rampup, flattop):
            assert merged["physics_randomization"]
        # The ramp source is exact upstream. The hot 51 MW source is split into
        # two exposed controls while preserving its shape and electron share.
        assert rampup["torax"]["sources"]["generic_heat"]["P_total"] == 20.0e6
        assert flattop["torax"]["sources"]["generic_heat"]["P_total"] == (
            23_314_285.714285713
        )
        rampup_actuators = {item["name"]: item for item in rampup["actuators"]}
        flattop_actuators = {item["name"]: item for item in flattop["actuators"]}
        assert rampup_actuators["P_nbi"]["init"] == 20.0e6
        assert flattop_actuators["P_nbi"]["init"] == 23_314_285.714285713
        # Phase deltas differ and win over base.
        assert rampup["torax"]["numerics"]["t_final"] == 100.0
        assert rampup["torax"]["numerics"]["fixed_dt"] == 0.1
        assert flattop["torax"]["numerics"]["t_final"] == 440.0
        assert flattop["torax"]["numerics"]["fixed_dt"] == 0.1
        assert "geometry_configs" in rampup["torax"]["geometry"]
        assert (
            flattop["torax"]["geometry"]["geometry_file"] == "iter_hybrid_ip105.eqdsk"
        )
        # scenario:/phase: metadata keys are stripped from the merged config.
        assert "scenario" not in rampup
        assert "phase" not in rampup


class RadiationBackendIndependenceTest:
    """Ohmic + brems + Mavrin impurity radiation are tokamak-owned, so every
    transport backend resolves the same source stack (closes the old ITER+BgB
    impurity-radiation omission)."""

    def test_hybrid_control_horizons_identical_across_backends(self):
        for backend in (
            "cgm",
            "qlknn",
            "tglfnn",
            "tglfnn_nr",
            "bohm_gyrobohm",
        ):
            rampup = _merge("iter/hybrid/rampup", backend)["torax"]["numerics"]
            flattop = _merge("iter/hybrid/flattop", backend)["torax"]["numerics"]
            assert (rampup["t_final"], rampup["fixed_dt"]) == (100.0, 0.1)
            assert (flattop["t_final"], flattop["fixed_dt"]) == (440.0, 0.1)

    def test_iter_source_stack_identical_across_backends(self):
        for backend in (
            "cgm",
            "qlknn",
            "tglfnn",
            "tglfnn_nr",
            "bohm_gyrobohm",
        ):
            src = _merge("iter/hybrid/flattop", backend)["torax"]["sources"]
            assert "ohmic" in src
            assert src["bremsstrahlung"]["use_relativistic_correction"] is True
            assert src["impurity_radiation"]["model_name"] == "mavrin_fit"

    def test_sparc_source_stack_identical_across_backends(self):
        for backend in ("cgm", "bohm_gyrobohm"):
            src = _merge("sparc/prd/flattop", backend)["torax"]["sources"]
            assert "ohmic" in src
            assert src["impurity_radiation"]["model_name"] == "mavrin_fit"

    def test_radiation_multiplier_randomization_tokamak_owned(self):
        # Moved from the surrogate backends to the tokamak layer, so it is
        # present even on bohm_gyrobohm which never declared it.
        for backend in ("cgm", "bohm_gyrobohm"):
            rand = _merge("iter/hybrid/rampdown", backend)["physics_randomization"]
            assert "sources.impurity_radiation.radiation_multiplier" in rand

    def test_step_keeps_lumped_radiation(self):
        # STEP's env-side P_in_scaled_flat_profile lump already includes
        # synchrotron; dropping backend-owned brems removes a double-count.
        src = _merge_env_and_backend(
            resolve_env("step"), resolve_backend("bohm_gyrobohm")
        )["torax"]["sources"]
        assert src["impurity_radiation"]["model_name"] == "P_in_scaled_flat_profile"
        assert "bremsstrahlung" not in src
        assert "cyclotron_radiation" not in src


class SawtoothMergeTest:
    """Sawtooth MHD is scenario-owned and statically budgeted."""

    def test_sawtooth_enabled_where_q_crosses_one(self):
        for scenario in (
            "iter/baseline",
            "iter/hybrid",
            "sparc/prd",
            "sparc/reduced_field",
        ):
            for phase in ("rampup", "flattop", "rampdown"):
                merged = _merge(f"{scenario}/{phase}")
                saw = merged["torax"]["mhd"]["sawtooth"]
                assert saw["trigger_model"] == {
                    "model_name": "simple",
                    "s_critical": 0.1,
                    "minimum_radius": 0.05,
                }
                assert saw["redistribution_model"] == {
                    "model_name": "simple",
                    "flattening_factor": 1.01,
                    "mixing_radius_multiplier": 1.1,
                }
                assert saw["crash_step_duration"] == 1.0e-5
                assert merged["stepping"]["max_event_substeps"] == 1

    def test_sawtooth_absent_where_q_stays_above_one(self):
        for phase in ("rampup", "flattop", "rampdown"):
            merged = _merge(f"iter/advanced/{phase}")
            assert "mhd" not in merged["torax"]
            assert "stepping" not in merged
        step = _merge_env_and_backend(
            resolve_env("step"), resolve_backend("bohm_gyrobohm")
        )
        assert "mhd" not in step["torax"]
        assert "stepping" not in step

    def test_tglfnn_nr_combines_solver_and_event_budgets(self):
        merged = _merge("iter/hybrid/flattop", "tglfnn_nr")
        assert merged["stepping"] == {
            "max_solver_substeps": 32,
            "max_event_substeps": 1,
        }

    def test_qlknn_proxy_disabled(self):
        # A QLKNN-only input clamp would double-count the physical crash model
        # and make q=1 handling backend-specific.
        transport = _merge("iter/hybrid/flattop", "qlknn")["torax"]["transport"]
        assert transport["q_sawtooth_proxy"] is False


class SparcMergeTest:
    def test_actuators_and_disruption_inherited_from_tokamak(self):
        # SPARC scenarios inherit the SPARC device interface (ICRF-scaled P_nbi,
        # higher Greenwald threshold) from tokamaks/sparc.yaml.
        for env in ("sparc/prd/rampup", "sparc/reduced_field/flattop"):
            m = _merge(env)
            acts = {a["name"]: a for a in m["actuators"]}
            assert acts["P_nbi"]["high"] == 25.0e6
            assert m["disruption"]["greenwald_threshold"] == 1.1

    def test_scenario_base_shared_across_phases_with_phase_overrides(self):
        rampup = _merge("sparc/prd/rampup")
        flattop = _merge("sparc/prd/flattop")
        # Base-only physics (pedestal, sources, constant CD fraction) reaches
        # every phase identically.
        for m in (rampup, flattop):
            assert m["torax"]["pedestal"]["T_i_ped"] == 4.2825268
            assert m["torax"]["sources"]["generic_heat"]["P_total"] == 11.0e6
            assert (
                m["torax"]["sources"]["generic_current"]["fraction_of_total_current"]
                == 0.1
            )
        # Phase deltas differ and win over base.
        assert rampup["torax"]["numerics"]["t_final"] == 10.0
        assert flattop["torax"]["numerics"]["t_final"] == 15.0
        assert "geometry_configs" in rampup["torax"]["geometry"]
        assert (
            flattop["torax"]["geometry"]["geometry_file"]
            == "references/sparc_prd_freegs_20221013.eqdsk"
        )
        # scenario:/phase: metadata keys are stripped from the merged config.
        assert "scenario" not in rampup
        assert "phase" not in rampup

    def test_reduced_field_uses_its_own_geometry(self):
        m = _merge("sparc/reduced_field/rampup")
        stems = {
            v["geometry_file"]
            for v in m["torax"]["geometry"]["geometry_configs"].values()
        }
        assert stems == {
            "sparc_reduced_field_ip010.eqdsk",
            "sparc_reduced_field_ip020.eqdsk",
            "sparc_reduced_field_ip057.eqdsk",
        }
