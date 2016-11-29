package org.mg.wekalib.test;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.Random;

import javax.swing.JFrame;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.evaluation.ThresholdCurve;
import weka.core.Instances;
import weka.gui.visualize.PlotData2D;
import weka.gui.visualize.ThresholdVisualizePanel;

public class PRCurveTest
{
	public static void main(String[] args) throws Exception
	{
		String arff = "/home/martin/data/weka/nominal/colic.arff";
		// load data
		Instances data = new Instances(new BufferedReader(new FileReader(arff)));
		data.setClassIndex(data.numAttributes() - 1);

		// train classifier
		Classifier cl = new NaiveBayes();
		Evaluation eval = new Evaluation(data);
		eval.crossValidateModel(cl, data, 10, new Random(1));

		// generate curve
		ThresholdCurve tc = new ThresholdCurve();
		int classIndex = 0;
		Instances result = tc.getCurve(eval.predictions(), classIndex);

		// plot curve
		ThresholdVisualizePanel vmc = new ThresholdVisualizePanel();
		PlotData2D tempd = new PlotData2D(result);

		// specify which points are connected
		boolean[] cp = new boolean[result.numInstances()];
		for (int n = 1; n < cp.length; n++)
			cp[n] = true;
		tempd.setConnectPoints(cp);
		// add plot
		vmc.addPlot(tempd);

		// We want a precision-recall curve
		vmc.setXIndex(result.attribute("Recall").index());
		vmc.setYIndex(result.attribute("Precision").index());

		// Make window with plot but don't show it
		JFrame jf = new JFrame();
		jf.setSize(500, 400);
		jf.getContentPane().add(vmc);
		jf.pack();

		jf.setVisible(true);

		//		// Save to file specified as second argument (can use any of
		//		// BMPWriter, JPEGWriter, PNGWriter, PostscriptWriter for different formats)
		//		JComponentWriter jcw = new JPEGWriter(vmc.getPlotPanel(), new File(args[1]));
		//		jcw.toOutput();
		//		System.exit(1);
	}
}
