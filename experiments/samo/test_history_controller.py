"""Unit tests for conservative history controller."""
import tempfile
import unittest
from pathlib import Path
from experiments.samo.history_controller import HistoryController,load_history

class HistoryTests(unittest.TestCase):
    def test_empty_history_is_neutral(self):
        c=HistoryController([])
        self.assertEqual(c.update(5,[1,1,1,1],[.1,.1,.1]),[.1,.1,.1])
    def test_bounds_and_rate_limit(self):
        ex=[([1.,2.,3.,.5],[.3,.02,.2],.8)]*8
        c=HistoryController(ex,warmup=0)
        w=c.update(2,[1,2,3,.5],[.1,.1,.1])
        self.assertAlmostEqual(w[0],.12)
        self.assertAlmostEqual(w[1],.08)
    def test_load_future_outcome(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"epochs.csv"
            p.write_text("""train_fusion_loss,train_img_loss,train_gene_loss,train_rank_loss,weight_img,weight_gene,weight_rank,val_ci
1,2,3,.5,.1,.1,.1,.6
.9,1.9,2.9,.4,.2,.1,.1,.7
""")
            x=load_history([p],horizon=1)
            self.assertEqual(len(x),1)
            self.assertAlmostEqual(x[0][2],.7)

if __name__=="__main__":
    unittest.main()
