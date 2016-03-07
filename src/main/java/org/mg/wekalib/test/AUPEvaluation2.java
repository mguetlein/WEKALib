package org.mg.wekalib.test;

import java.awt.Dimension;
import java.awt.GridLayout;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import javax.swing.JPanel;

import org.mg.javalib.util.StringUtil;
import org.mg.javalib.util.SwingUtil;
import org.mg.wekalib.evaluation.PredictionUtil;
import org.mg.wekalib.evaluation.PredictionUtil.ClassificationMeasure;
import org.mg.wekalib.evaluation.PredictionUtilPlots;
import org.mg.wekalib.evaluation.Predictions;

public class AUPEvaluation2
{
	static HashMap<ClassificationMeasure, Double> reference = new HashMap<>();

	static int x, y;

	static int seriesIdx = 0;

	static List<JPanel> panels = new ArrayList<>();
	static int rows = 0;
	static int cols = 2;

	static Predictions refP = null;

	public static void main(String args[]) throws Exception
	{
		eval2("11011111110111000000010000000000000000000000010000000000000000100", true);
		eval2("11011111110111000000010000000000000000000000010000000000001000000", false);
		eval2("11011111110111000000010000000000000000000100000000000000000000100", false);
		eval2("11011111110111000100000000000000000000000000010000000000000000100", false);
		eval2("11111101110111000000010000000000000000000000010000000000000000100", false);
		plot();
	}

	public static void eval2(String bitstring, boolean ref) throws Exception
	{
		if (ref)
			System.out.println("(" + "abcde".charAt(seriesIdx) + ") Reference:");
		else
			System.out.println("(" + "abcde".charAt(seriesIdx) + ") Improvement:");
		System.out.println(bitstring);
		Predictions pred = PredictionUtil.fromBitString(bitstring);
		if (ref)
			refP = pred;

		//		System.out.println(ArrayUtil.toString(clazzes));
		//		for (double[] ds : predictions)
		//			System.out.println(ArrayUtil.toString(ds));

		//		System.out
		//				.println("accuracy  " + StringUtil.formatDouble(PredictionUtil.accuracy(pred), 2));

		String p;
		double d;
		double diff = 0;

		for (ClassificationMeasure m : new ClassificationMeasure[] { ClassificationMeasure.AUC,
				ClassificationMeasure.AUPRC, ClassificationMeasure.EnrichmentFactor5,
				ClassificationMeasure.BEDROC20, ClassificationMeasure.BEDROC100 })
		{
			p = m.shortName();
			d = PredictionUtil.getClassificationMeasure(pred, m, 1);

			if (ref)
				reference.put(m, d);
			else
				diff = d - reference.get(m);

			int dec = 3;
			if (m == ClassificationMeasure.EnrichmentFactor5)
				dec = 1;

			String s = StringUtil.formatDouble(d, dec);
			if (!ref)
			{
				if (diff >= 0)
					s += " +" + StringUtil.formatDouble(diff, dec);
				else
					s += " " + StringUtil.formatDouble(diff, dec);
			}
			System.out.print(StringUtil.concatWhitespace(p + " " + s, 23));

			if (!ref && (m == ClassificationMeasure.AUC || m == ClassificationMeasure.AUPRC))
			{
				JPanel panel = PredictionUtilPlots.getPlot(m, 1, pred, refP);
				panel.setPreferredSize(new Dimension(220, 220));
				panel.setOpaque(false);
				//				String fName = null;
				//				int i = 0;
				//				while (fName == null || new File(fName).exists())
				//					fName = "/tmp/plot/" + m + "-" + i++ + ".png";
				//				SwingUtil.toFile(fName, panel, panel.getPreferredSize());
				panels.add(panel);
				//ThreadUtil.sleep(100);
			}
		}
		System.out.println("\n");
		seriesIdx++;
		//		System.out.println("f-measure " + StringUtil.formatDouble(eval.fMeasure(1)));
		//		System.out.println("recall    " + StringUtil.formatDouble(eval.recall(1)));
		//		System.out.println("precision " + StringUtil.formatDouble(eval.precision(1)));

		//		System.out.println();

	}

	public static void plot()
	{
		JPanel p = new JPanel(new GridLayout(rows, cols, 5, 5));
		for (JPanel pa : panels)
			p.add(pa);
		SwingUtil.showInFrame(p);
		//		SwingUtil.toFile("/tmp/aup-" + auc + ".png", p, p.getPreferredSize());
	}
}
