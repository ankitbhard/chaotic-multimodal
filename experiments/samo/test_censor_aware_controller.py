import unittest
from experiments.samo.censor_aware_controller import CensorAwareController, survival_information

class TestV3(unittest.TestCase):
    def test_pairs_censored(self):
        x=survival_information([1,2,3],[1,0,1])
        self.assertAlmostEqual(x["event_rate"],2/3)
        self.assertAlmostEqual(x["pair_density"],1/3)
    def test_exact_zero_baseline(self):
        c=CensorAwareController("ra_samo_v3_0")
        self.assertEqual(c.state.weights,(0.,0.,0.))
    def test_all_variants_construct(self):
        for i in range(4):
            c=CensorAwareController(f"ra_samo_v3_{i}",warmup=0,switch_margin=0,min_dwell=0)
            state=c.update(0,[2.,2.,2.,2.],{"event_rate":.4,"pair_density":.2})
            self.assertIn(state.mode,("cox","aux","rank"))
            self.assertTrue(all(v>=0 for v in state.weights))
    def test_rank_mode_exact_gates(self):
        c=CensorAwareController("ra_samo_v3_0",warmup=0,switch_margin=0,min_dwell=0)
        c.update(0,[2.,2.,2.,2.],{})
        state=c.update(1,[1.9,2.,2.,.5],{})
        self.assertEqual(state.mode,"rank")
        self.assertEqual(state.weights[:2],(0.,0.))
if __name__=="__main__":unittest.main()
