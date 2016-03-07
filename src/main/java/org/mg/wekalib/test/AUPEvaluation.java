package org.mg.wekalib.test;

import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.GridLayout;
import java.io.File;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import javax.swing.JPanel;

import org.apache.commons.lang3.ArrayUtils;
import org.mg.javalib.util.ArrayUtil;
import org.mg.javalib.util.StringUtil;
import org.mg.javalib.util.SwingUtil;
import org.mg.wekalib.data.ArffWritable;
import org.mg.wekalib.data.ArffWriter;
import org.mg.wekalib.evaluation.alt.ClassificationMeasure;
import org.mg.wekalib.evaluation.alt.ExtendedEvaluation;
import org.mg.wekalib.evaluation.alt.Measure;

import weka.classifiers.Classifier;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.evaluation.ThresholdCurve;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import weka.gui.visualize.Plot2D;
import weka.gui.visualize.PlotData2D;
import weka.gui.visualize.ThresholdVisualizePanel;

public class AUPEvaluation
{
	static ClassificationMeasure[] measures = new ClassificationMeasure[] {
			ClassificationMeasure.AUC, ClassificationMeasure.AUP, ClassificationMeasure.ER5 };

	//static ClassificationMeasure[] measures = new ClassificationMeasure[] { ClassificationMeasure.AUC };

	static HashMap<Measure, Double> reference = new HashMap<>();

	static int x, y;

	static int seriesIdx = 0;

	static List<JPanel> panels = new ArrayList<>();
	static int rows;
	static int cols;

	public static void main(String args[]) throws Exception
	{
		//		Instances data = new Instances(new FileReader("/home/martin/data/weka/nominal/breast-cancer.arff"));
		//		data.setClassIndex(data.numAttributes() - 1);
		//
		//		// train classifier
		//		Classifier cl = new NaiveBayes();
		//		Evaluation eval = new Evaluation(data);
		//		eval.crossValidateModel(cl, data, 10, new Random(1));
		//
		//		// generate curve
		//		ThresholdCurve tc = new ThresholdCurve();
		//		int classIndex = 0;
		//		Instances result = tc.getCurve(eval.predictions(), classIndex);
		//
		//		// plot curve
		//		ThresholdVisualizePanel vmc = new ThresholdVisualizePanel();
		//		vmc.setROCString("(Area under ROC = " + Utils.doubleToString(tc.getROCArea(result), 4) + ")");
		//		vmc.setName(result.relationName());
		//		PlotData2D tempd = new PlotData2D(result);
		//		tempd.setPlotName(result.relationName());
		//		tempd.addInstanceNumberAttribute();
		//		// specify which points are connected
		//		boolean[] cp = new boolean[result.numInstances()];
		//		for (int n = 1; n < cp.length; n++)
		//			cp[n] = true;
		//		tempd.setConnectPoints(cp);
		//		// add plot
		//		vmc.addPlot(tempd);
		//		//vmc.setXIndex(1);
		//
		//		// display curve
		//		String plotName = vmc.getName();
		//		final javax.swing.JFrame jf = new javax.swing.JFrame("Weka Classifier Visualize: " + plotName);
		//		jf.setSize(500, 400);
		//		jf.getContentPane().setLayout(new BorderLayout());
		//		jf.getContentPane().add(vmc, BorderLayout.CENTER);
		//		jf.addWindowListener(new java.awt.event.WindowAdapter()
		//		{
		//			public void windowClosing(java.awt.event.WindowEvent e)
		//			{
		//				jf.dispose();
		//			}
		//		});
		//		jf.setVisible(true);
		//		if (true == true)
		//			return;

		//		String actual, predicted;
		//
		//		actual/*   */= "1,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0";
		//		predicted/**/= "x,y,9,8,7,6,5,4,3,2,8,7,7,6,6,6,5,5,5,5,4,4,4,4,4,3,3,3,3,3,3,2,2,2,2,2,2,2,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0";
		//		x = 9;
		//		y = 0;
		//		eval(actual, predicted);
		//
		//		x = 6;
		//		y = 0;
		//		eval(actual, predicted);
		//
		//		x = 9;
		//		y = 8;
		//		eval(actual, predicted);

		panels.clear();
		rows = 0;
		cols = 2;

		//		eval2("11111101111110110001000000000000000000100000000000000000000000", true);
		//		eval2("11111101111110110001000000000000000000000100000000000000000000", false);
		//		eval2("11111101111110110000001000000000000000100000000000000000000000", false);
		//		eval2("11111101110111110001000000000000000000100000000000000000000000", false);
		//		eval2("11101111111110110001000000000000000000100000000000000000000000", false);

		//		eval2("1101000", true);

		eval2("11011111110111000000010000000000000000000000010000000000000000100", true);
		eval2("11011111110111000000010000000000000000000000010000000000001000000", false);
		eval2("11011111110111000000010000000000000000000100000000000000000000100", false);
		eval2("11011111110111000100000000000000000000000000010000000000000000100", false);
		eval2("11111101110111000000010000000000000000000000010000000000000000100", false);

		//			eval2("01111111111110110001000000000000000000100000000000000000000000");
		//		rows++;
		//
		plot();
		//		System.out.println("\n");
	}

	public static void eval2(String rank, String msg, boolean ref) throws Exception
	{
		System.out.print(msg + ": ");
		eval2(rank, ref);
	}

	public static void eval2(String rank, boolean ref) throws Exception
	{
		if (ref)
			System.out.println("(" + "abcde".charAt(seriesIdx) + ") Reference:");
		else
			System.out.println("(" + "abcde".charAt(seriesIdx) + ") Improvement:");
		System.out.println(rank);
		final String clazzes[] = ArrayUtil.toStringArray(ArrayUtils.toObject(rank.toCharArray()));
		final List<double[]> predictions = new ArrayList<>();
		double step = 1 / (double) (clazzes.length - 1);
		for (int i = 0; i < clazzes.length; i++)
		{
			double p = 1 - (i * step);
			//			System.out.println(p);
			predictions.add(new double[] { 1 - p, p });
		}
		eval(clazzes, predictions, ref);
	}

	public static void eval(String actual, String predicted, boolean ref) throws Exception
	{
		final String clazzes[] = actual.split(",");
		final List<double[]> predictions = new ArrayList<>();
		for (String p : predicted.split(","))
		{
			double p1;
			if (p.equals("x"))
				p1 = x * 0.1;
			else if (p.equals("y"))
				p1 = y * 0.1;
			else
				p1 = Double.parseDouble("." + p);
			predictions.add(new double[] { 1.0 - p1, p1 });
		}
		eval(clazzes, predictions, ref);
	}

	public static void eval(final String[] clazzes, final List<double[]> predictions, boolean ref)
			throws Exception
	{
		System.out.println(ArrayUtil.toString(clazzes));
		for (double[] ds : predictions)
			System.out.println(ArrayUtil.toString(ds));

		Instances inst = ArffWriter.toInstances(new ArffWritable()
		{
			@Override
			public boolean isSparse()
			{
				return false;
			}

			//			@Override
			//			public boolean isInstanceWithoutAttributeValues(int instance)
			//			{
			//				return false;
			//			}

			@Override
			public String getRelationName()
			{
				return "bla";
			}

			@Override
			public int getNumInstances()
			{
				return clazzes.length;
			}

			@Override
			public int getNumAttributes()
			{
				return 1;
			}

			@Override
			public String getMissingValue(int attribute)
			{
				return null;
			}

			@Override
			public String[] getAttributeDomain(int attribute)
			{
				return new String[] { "0", "1" };
			}

			@Override
			public String getAttributeValue(int instance, int attribute) throws Exception
			{
				return clazzes[instance];
			}

			@Override
			public double getAttributeValueAsDouble(int instance, int attribute) throws Exception
			{
				return clazzes[instance].equals("1") ? 1.0 : 0.0;
			}

			@Override
			public String getAttributeName(int attribute)
			{
				return "clazz";
			}

			@Override
			public List<String> getAdditionalInfo()
			{
				return null;
			}
		});
		inst.setClassIndex(0);

		Classifier cl = new Classifier()
		{
			int predCount = 0;

			@Override
			public Capabilities getCapabilities()
			{
				return null;
			}

			@Override
			public double[] distributionForInstance(Instance instance) throws Exception
			{
				return predictions.get(predCount++);
			}

			@Override
			public double classifyInstance(Instance instance) throws Exception
			{
				throw new RuntimeException();
			}

			@Override
			public void buildClassifier(Instances data) throws Exception
			{
			}
		};

		Evaluation eval = new ExtendedEvaluation(inst);
		eval.evaluateModel(cl, inst);
		double times = 1;

		//		eval.setMetricsToDisplay(ArrayUtil.toList(Evaluation.BUILT_IN_EVAL_METRICS));
		//		System.out.println("num       " + eval.numInstances());
		System.out.println("accuracy  " + StringUtil.formatDouble(0.01 * eval.pctCorrect(), 2));

		String p;
		double d;
		double diff = 0;

		for (Measure m : measures)
		{
			p = m.toString();
			d = m.getValue(eval);

			d *= times;
			if (ref)
				reference.put(m, d);
			else
				diff = d - reference.get(m);

			int dec = 3;
			if (m == ClassificationMeasure.ER5)
				dec = 1;

			String s = StringUtil.formatDouble(d, dec);
			//			if (diff != 0)
			//			{
			//\u0394
			if (!ref)
			{
				if (diff >= 0)
					s += " +" + StringUtil.formatDouble(diff, dec);
				else
					s += " " + StringUtil.formatDouble(diff, dec);
			}
			//			}
			System.out.print(StringUtil.concatWhitespace(p + " " + s, 23));

			if (m == ClassificationMeasure.AUC || m == ClassificationMeasure.AUP)
			{
				ThresholdCurve tc = new ThresholdCurve();
				// method visualize
				ThresholdVisualizePanel vmc = new ThresholdVisualizePanel()
				{
					@Override
					public void addPlot(PlotData2D newPlot) throws Exception
					{
						m_plot = new PlotPanel()
						{
							@Override
							public void addPlot(PlotData2D newPlot) throws Exception
							{
								m_plot2D = new Plot2D()
								{
									public void addPlot(PlotData2D newPlot) throws Exception
									{
										m_axisColour = Color.BLACK;
										super.addPlot(newPlot);
									};
								};
								m_plot2D.setBackground(Color.WHITE);
								this.add(m_plot2D, BorderLayout.CENTER);
								super.addPlot(newPlot);
							}
						};
						super.addPlot(newPlot);
					}
				};
				vmc.setROCString(
						"(Area under ROC = " + Utils.doubleToString(eval.areaUnderPRC(1), 4) + ")");
				vmc.setName("name");

				//				ArrayList<Prediction> preds = eval.predictions();
				//				Collections.sort(preds, new Comparator<Prediction>()
				//				{
				//					@Override
				//					public int compare(Prediction o1, Prediction o2)
				//					{
				//						return new Double(o2.actual()).compareTo(o1.actual());
				//					}
				//				});

				Instances curve = tc.getCurve(eval.predictions());

				//				Instances curveX = new Instances(curve);
				//				while (curveX.numInstances() > 0)
				//					curveX.remove(0);
				//				for (int i = 0; i < curve.numInstances(); i++)
				//					curveX.add(curve.get(curve.numInstances() - (1 + i)));
				//				curve = curveX;

				List<String> attributes = new ArrayList<>();
				for (int i = 0; i < curve.numAttributes(); i++)
				{
					attributes.add(curve.attribute(i).name());
				}
				System.err.println(attributes);
				//				System.err.println(curve.numInstances());
				//				System.err.println(curve);

				String yAttr = "";
				String xAttr = "";
				if (m == ClassificationMeasure.AUP)
				{
					xAttr = "Recall";
					yAttr = "Precision";
				}
				else if (m == ClassificationMeasure.AUC)
				{
					xAttr = "False Positive Rate";
					yAttr = "Recall";//True Positive Rate";
				}
				System.err.println("X " + curve.attribute(xAttr).name());
				for (int i = 0; i < curve.numInstances(); i++)
					System.err.println(i + ": " + curve.get(i).value(curve.attribute(xAttr)));
				System.err.println("Y " + curve.attribute(yAttr).name());
				for (int i = 0; i < curve.numInstances(); i++)
					System.err.println(i + ": " + curve.get(i).value(curve.attribute(yAttr)));

				PlotData2D tempd = new PlotData2D(curve);
				tempd.setPlotName("name2");
				tempd.addInstanceNumberAttribute();
				tempd.setCustomColour(Color.BLACK);

				//				String axisKey = thisClass + ".axisColour";
				//			      String backgroundKey = thisClass + ".backgroundColour";

				//			      VisualizeUtils.VISUALIZE_PROPERTIES
				//			        .getProperty(axisKey);

				// specify which points are connected
				boolean[] cp = new boolean[curve.numInstances()];
				for (int n = 1; n < cp.length; n++)
					cp[n] = true;
				tempd.setConnectPoints(cp);

				// add plot
				vmc.addPlot(tempd);

				//				vmc.setColourIndex(0);

				if (m == ClassificationMeasure.AUP)
				{
					cp[cp.length - 1] = false;
				}
				else if (m == ClassificationMeasure.AUC)
				{
				}

				vmc.setYIndex(attributes.indexOf(yAttr) + 1);
				vmc.setXIndex(attributes.indexOf(xAttr) + 1);

				//SwingUtil.showInDialog(vmc);

				JPanel panel = vmc.getPlotPanel(); // vmc;
				panel.setPreferredSize(new Dimension(220, 220));
				panel.setOpaque(false);
				String fName = null;
				int i = 0;
				while (fName == null || new File(fName).exists())
					fName = "/tmp/plot/" + m + "-" + i++ + ".png";
				SwingUtil.toFile(fName, panel, panel.getPreferredSize());
				panels.add(panel);
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
