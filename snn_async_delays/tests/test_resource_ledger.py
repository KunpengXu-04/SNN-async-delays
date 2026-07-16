import os
import sys
import unittest
from functools import partial

from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data.boolean_dataset import MultiQueryDataset, ExhaustiveFixedOperationQueryDataset
from data.encoding import encode_sequential_trial
from snn.model import SNNSimultaneousModel
from train.eval import evaluate_simultaneous
from utils.resource_ledger import static_resource_ledger, dynamic_resource_ledger


def model(mode="late_window", output=False, layers=1):
    sizes = [6] if layers == 1 else [4, 3]
    return SNNSimultaneousModel(
        n_queries=2,
        n_hidden=sizes[0],
        win_len=8,
        read_len=4,
        d_max=8,
        train_mode="weights_only",
        fixed_delay_value=0.0,
        n_input_channels=2,
        readout_type="linear",
        num_hidden_layers=layers,
        hidden_sizes=sizes,
        use_output_spikes=output,
        n_output_neurons=2 if output else None,
        observation_mode=mode,
    )


class ResourceLedgerTests(unittest.TestCase):
    def test_static_counts_for_one_layer_count_readout(self):
        late = static_resource_ledger(model("late_window"))
        all_time = static_resource_ledger(model("all_time"))
        binned = static_resource_ledger(model("time_binned"))

        self.assertEqual(late["delayed_synapses_total"], 12)
        self.assertEqual(late["delay_buffer_elements_per_sample"], 18)
        self.assertEqual(late["neuron_updates_per_trial"], 72)
        self.assertEqual(late["dense_synapse_macs_per_trial"], 144)
        self.assertEqual(late["decoder_weight_macs_per_trial"], 12)
        self.assertEqual(late["readout_accumulation_adds_per_trial"], 24)
        self.assertEqual(all_time["readout_accumulation_adds_per_trial"], 72)
        self.assertEqual(all_time["decoder_weight_macs_per_trial"], 12)
        self.assertEqual(binned["decoder_weight_macs_per_trial"], 36)
        self.assertEqual(binned["readout_feature_dim"], 18)

    def test_shared_window_decoder_is_charged_once_in_storage_and_per_window_in_compute(self):
        shared = SNNSimultaneousModel(
            n_queries=3, n_hidden=6, win_len=8, read_len=12,
            n_input_channels=2, train_mode="weights_only", fixed_delay_value=0.0,
            readout_type="linear", observation_mode="windowed_shared",
            output_window_len=4,
        )
        ledger = static_resource_ledger(shared)
        self.assertEqual(ledger["decoder_storage_elements"], 7)
        self.assertEqual(ledger["decoder_weight_macs_per_trial"], 18)
        self.assertEqual(ledger["readout_accumulation_adds_per_trial"], 72)

    def test_dynamic_event_fanout_counts(self):
        count_model = model()
        dynamic = dynamic_resource_ledger(
            count_model, mean_input_spikes=3, mean_hidden1_spikes=5
        )
        self.assertEqual(dynamic["mean_synaptic_events_input_hidden"], 18)
        self.assertEqual(dynamic["mean_synaptic_events_total"], 18)

        output_model = model(output=True)
        output_static = static_resource_ledger(output_model)
        self.assertEqual(output_static["decoder_storage_elements"], 24)
        output_dynamic = dynamic_resource_ledger(
            output_model,
            mean_input_spikes=3,
            mean_hidden1_spikes=5,
            mean_output_spikes=1,
        )
        self.assertEqual(output_dynamic["mean_synaptic_events_hidden_output"], 10)
        self.assertEqual(output_dynamic["mean_synaptic_events_total"], 28)

        two_layer = model(layers=2)
        two_dynamic = dynamic_resource_ledger(
            two_layer,
            mean_input_spikes=3,
            mean_hidden1_spikes=5,
            mean_hidden2_spikes=2,
        )
        self.assertEqual(two_dynamic["mean_synaptic_events_input_hidden"], 12)
        self.assertEqual(two_dynamic["mean_synaptic_events_hidden_hidden"], 15)
        self.assertEqual(two_dynamic["mean_synaptic_events_total"], 27)

    def test_evaluation_embeds_measured_resource_ledger(self):
        test_model = model("all_time")
        dataset = MultiQueryDataset(
            K=2, n_samples=16, same_op=True, op_name="NAND", seed=9
        )
        loader = DataLoader(dataset, batch_size=8, shuffle=False)
        cfg = {
            "win_len": 8,
            "read_len": 4,
            "r_on": 400.0,
            "r_off": 10.0,
            "dt": 1.0,
            "n_ops": 0,
        }
        encode = partial(
            encode_sequential_trial,
            encoding_mode="burst",
            burst_n_spikes_on=2,
            burst_n_spikes_off=1,
            burst_jitter_ms=0,
        )
        result = evaluate_simultaneous(
            test_model, loader, cfg, device="cpu", encode_fn=encode
        )
        ledger = result["resource_ledger"]
        self.assertEqual(ledger["schema_version"], "resource_ledger_v1")
        self.assertGreater(ledger["mean_input_spikes_per_trial"], 0)
        self.assertEqual(
            ledger["mean_synaptic_events_input_hidden"],
            ledger["mean_input_spikes_per_trial"] * 6,
        )
        self.assertEqual(ledger["observation_mode"], "all_time")
        self.assertIn("worst_query_balanced_accuracy", result)
        self.assertEqual(result["pooled_accuracy"], result["accuracy"])

    def test_opponent_evaluation_reports_interface_and_truth_table_metrics(self):
        test_model = SNNSimultaneousModel(
            n_queries=3, n_hidden=6, win_len=8, read_len=12,
            n_input_channels=6, train_mode="weights_only", fixed_delay_value=0.0,
            use_output_spikes=True, n_output_neurons=2,
            opponent_output_mode="shared_windowed", output_window_len=4,
            observation_mode="windowed_shared",
        )
        dataset = ExhaustiveFixedOperationQueryDataset(["XOR", "NAND", "NOR"])
        loader = DataLoader(dataset, batch_size=16, shuffle=False)
        cfg = {"win_len": 8, "read_len": 12, "r_on": 400.0, "r_off": 10.0,
               "dt": 1.0, "n_ops": 0}
        from data.encoding import encode_simultaneous_trial
        encode = partial(encode_simultaneous_trial, encoding_mode="burst",
                         burst_n_spikes_on=2, burst_n_spikes_off=1,
                         burst_jitter_ms=0)
        result = evaluate_simultaneous(test_model, loader, cfg, device="cpu",
                                       encode_fn=encode, return_trial_records=True)
        self.assertEqual(result["truth_table_patterns"], 64)
        self.assertEqual(len(result["trial_records"]), 64)
        self.assertIsNotNone(result["output_silent_rate"])
        self.assertEqual(len(result["per_query_hidden_window_spikes"]), 3)
        self.assertEqual(len(result["per_query_hidden_window_activity_fraction"]), 3)
        self.assertEqual(len(result["cross_target_balanced_accuracy_matrix"]), 3)


if __name__ == "__main__":
    unittest.main()
