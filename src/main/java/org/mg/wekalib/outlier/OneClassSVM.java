package org.mg.wekalib.outlier;

import java.awt.Color;
import java.io.FileReader;
import java.util.Arrays;
import java.util.Collections;
import java.util.Random;
import java.util.Vector;

import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.renderer.xy.XYLineAndShapeRenderer;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;
import org.mg.javalib.freechart.HistogramPanel;
import org.mg.javalib.util.CountedSet;
import org.mg.javalib.util.ScreenUtil;
import org.mg.javalib.util.StringUtil;
import org.mg.javalib.util.SwingUtil;
import org.mg.wekalib.data.InstanceUtil;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.functions.LibSVM;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SelectedTag;
import weka.core.Utils;

/**
 * unsupervised
 * 
 * returns prob of beeing inside
 * 
 * prob is created from nu values
 * 
 * @author martin
 *
 */
public class OneClassSVM extends AbstractClassifier
{
	private static final long serialVersionUID = 5L;

	private LibSVM[] svms;

	public static final double DEFAULT_MAX_OUTLIERS = 0.1;

	private double maxOutliersInTraining = DEFAULT_MAX_OUTLIERS;

	public static final int DEFAULT_NUM_STEPS = 3;

	private int intermediateSteps = DEFAULT_NUM_STEPS;

	public static final int DEFAULT_KERNEL_TYPE = LibSVM.KERNELTYPE_RBF;

	private int kernelType = DEFAULT_KERNEL_TYPE;

	public static final double DEFAULT_GAMMA = 0.2;

	private double gamma = DEFAULT_GAMMA;

	public static final int DEFAULT_DEGREE = 3;

	private int degree = DEFAULT_DEGREE;

	@Override
	public String[] getOptions()
	{
		Vector<String> result = new Vector<String>();
		result.add("-S");
		result.add("" + getIntermediateSteps());
		result.add("-M");
		result.add("" + getMaxOutliersInTraining());
		result.add("-K");
		result.add("" + getKernelType());
		result.add("-G");
		result.add("" + getGamma());
		result.add("-D");
		result.add("" + getDegree());
		Collections.addAll(result, super.getOptions());
		return result.toArray(new String[result.size()]);
	}

	@SuppressWarnings("unchecked")
	private static <T> T parseNumber(String[] options, char c, T defaultValue) throws Exception
	{
		String tmpStr;
		tmpStr = Utils.getOption(c, options);
		T result;
		if (tmpStr.length() != 0)
		{
			if (defaultValue instanceof Double)
				result = (T) Double.valueOf(tmpStr);
			else if (defaultValue instanceof Integer)
				result = (T) Integer.valueOf(tmpStr);
			else
				throw new IllegalArgumentException();
		}
		else
			result = defaultValue;
		return result;
	}

	public void setOptions(String[] options) throws Exception
	{
		intermediateSteps = parseNumber(options, 'S', DEFAULT_NUM_STEPS);
		maxOutliersInTraining = parseNumber(options, 'M', DEFAULT_MAX_OUTLIERS);
		kernelType = parseNumber(options, 'K', DEFAULT_KERNEL_TYPE);
		gamma = parseNumber(options, 'G', DEFAULT_GAMMA);
		degree = parseNumber(options, 'D', DEFAULT_DEGREE);
		super.setOptions(options);
		Utils.checkForRemainingOptions(options);
	}

	/**
	 * the more steps the finer is the probability that is returned
	 * 0 : only one svm with nu = maxOutliersInTraining (0.1)
	 *   -> prob values 0 and 1
	 *   
	 * 3 : 4 svms with nu values: 0.025, 0.05, 0.075, 0.1 
	 *   -> prob values 0, 0.25, 0.5, 0.75, and 1.0
	 */
	public void setIntermediateSteps(int intermediateSteps)
	{
		this.intermediateSteps = intermediateSteps;
	}

	public void setMaxOutliersInTraining(double maxOutliersInTraining)
	{
		this.maxOutliersInTraining = maxOutliersInTraining;
	}

	public int getIntermediateSteps()
	{
		return intermediateSteps;
	}

	public double getMaxOutliersInTraining()
	{
		return maxOutliersInTraining;
	}

	public int getKernelType()
	{
		return kernelType;
	}

	public double getGamma()
	{
		return gamma;
	}

	public void setGamma(double gamma)
	{
		this.gamma = gamma;
	}

	/**
	 * see {@link LibSVM#TAGS_KERNELTYPE}
	 * 
	 * @param kernelType
	 */
	public void setKernelType(int kernelType)
	{
		this.kernelType = kernelType;
	}

	public int getDegree()
	{
		return degree;
	}

	public void setDegree(int degree)
	{
		this.degree = degree;
	}

	@Override
	public void buildClassifier(Instances data) throws Exception
	{
		if (data.classIndex() == -1
				|| (data.classAttribute().isNominal() && data.classAttribute().numValues() > 1))
		{
			if (data.classIndex() != -1)
				data = InstanceUtil.stripAttributes(data, Arrays.asList(data.classAttribute()));
			String vals[] = new String[data.numInstances()];
			Arrays.fill(vals, "same-class");
			InstanceUtil.attachNominalAttribute(data, "new-class", Arrays.asList("same-class"),
					Arrays.asList(vals), true);
			data.setClassIndex(data.numAttributes() - 1);
		}

		svms = new LibSVM[intermediateSteps + 1];
		for (int i = 0; i < svms.length; i++)
		{
			double nu = (i + 1) / ((double) intermediateSteps + 1) * maxOutliersInTraining;
			System.err.println("[" + i + "] train svm with nu " + nu);
			LibSVM svm = new LibSVM();
			svm.setSVMType(new SelectedTag(2, LibSVM.TAGS_SVMTYPE));
			svm.setKernelType(new SelectedTag(kernelType, LibSVM.TAGS_KERNELTYPE));
			svm.setGamma(gamma);
			svm.setDegree(degree);
			svm.setNu(nu);
			svm.setShrinking(false);
			svm.buildClassifier(data);
			svms[i] = svm;
		}
	}

	/**
	 * computes the probability for being no outlier
	 * this is not really a prob, just a ranking
	 * created by different values for nu
	 * 
	 * @throws Exception 
	*/
	public double pInside(Instance instance) throws Exception
	{
		Double nu = null;
		for (int i = 0; i < svms.length; i++)
		{
			// start at lowest nu values, were we get the lowest amount of outliers
			double d = svms[i].classifyInstance(instance);
			if (Double.isNaN(d)) // classified as outlier
			{
				nu = svms[i].getNu();
				break;
			}
		}
		// scale nu values to p-inside
		// nu value == null (never outlier) -> p-inside 1.0
		// lowest nu value -> p-inside = 0
		// trick replace null for scaling
		// nu:   0.025, 0.05, 0.075, 0.1, null
		// prob: 0.0    0.25  0.5    0.75 1.0
		// replace null with 0.125 (maxNu)
		double maxNu = (intermediateSteps + 2) / ((double) intermediateSteps + 1)
				* maxOutliersInTraining;
		double minNu = 1 / ((double) intermediateSteps + 1) * maxOutliersInTraining;
		double deltaNu = maxOutliersInTraining;
		// System.err.println("[ " + minNu + " - " + maxNu + " ] delta: " + deltaNu);

		if (nu == null)
			nu = maxNu;
		double pInside = (nu - minNu) / deltaNu;
		return pInside;
	}

	/*
	 * there two possible use cases
	 * 
	 * [1] class values are inside and outside, unsure compounds with prob 0.5 are compounds in between
	 * [2] class is only inside, the more unsure, the more outside
	 * 
	 * here we go for [2] because we want a confidence ranking
	 */

	/**
	 * computes the probability for being NO outlier
	 * this is not really a prob, just a ranking
	 * created by different values for nu
	 * 
	 * converting this probability to the way weka is storing confidence
	 * 
	 * pInside=0 := distribution 0.5 for class 1
	 * pInside=1 := distribution 1 for class 1
	 * class0 = 1 - class1
	 */
	@Override
	public double[] distributionForInstance(Instance instance) throws Exception
	{
		//return new double[] { 0D, 1D };

		double pInside = pInside(instance);
		//		return new double[] { 1.0 - pInside, pInside };

		double positive = pInside * 0.5 + 0.5;
		return new double[] { 1.0 - positive, positive };
	}

	public static void main(String[] args) throws Exception
	{
		Instances inst = new Instances(new FileReader("/home/martin/data/weka/no-class/iris.arff"));
		inst.setClassIndex(inst.numAttributes() - 1);
		OneClassSVM clazz = new OneClassSVM();
		clazz.setIntermediateSteps(0);

		System.out.println("instances " + inst.numInstances() + "\n");
		for (double maxOutliersInTraining : new double[] { 0.1, 0.25, 0.5 })
		{
			clazz.setMaxOutliersInTraining(maxOutliersInTraining);
			clazz.buildClassifier(inst);
			int numOutliers = 0;
			for (Instance i : inst)
			{
				if (clazz.pInside(i) == 0)
					numOutliers++;
				else if (clazz.pInside(i) != 1)
					throw new IllegalStateException();
			}
			System.out.println(numOutliers + " outliers with max-outliers-in-training: "
					+ maxOutliersInTraining + "\n");
		}

		System.out.println("----------------------------");

		clazz.setMaxOutliersInTraining(0.2);
		for (int steps : new int[] { 0, 1, 2, 3 })
		{
			clazz.setIntermediateSteps(steps);
			clazz.buildClassifier(inst);
			DescriptiveStatistics stats = new DescriptiveStatistics();
			CountedSet<String> set = new CountedSet<>();
			for (Instance i : inst)
			{
				double p = clazz.pInside(i);
				stats.addValue(p);
				set.add(p + "");

				//				if (steps == 3)
				//				{
				//					System.out.println(clazz.pInside(i) + " "
				//							+ Arrays.toString(clazz.distributionForInstance(i)));
				//				}
			}
			System.out.println("prob-values with " + steps + " steps " + stats.getMean() + " +- "
					+ stats.getStandardDeviation() + " values: " + set.toString() + "\n");
		}

		System.out.println("----------------------------");

		Instances instX = new Instances(
				new FileReader("/home/martin/data/weka/nominal/CPDBAS_Mouse.arff"));
		instX.setClassIndex(instX.numAttributes() - 1);

		//		LOF lof = new LOF();
		//		Instances instCopy = InstanceUtil.stripAttributes(instX,
		//				Arrays.asList(instX.classAttribute()));
		//		lof.setInputFormat(instCopy);
		//		Instances lofOut = Filter.useFilter(instCopy, lof);

		OneClassSVM oneClass1 = new OneClassSVM();
		oneClass1.intermediateSteps = 5;
		oneClass1.setMaxOutliersInTraining(0.66);
		oneClass1.setKernelType(LibSVM.KERNELTYPE_POLYNOMIAL);
		oneClass1.setDegree(2);
		oneClass1.buildClassifier(instX);

		OneClassSVM oneClass2 = new OneClassSVM();
		oneClass2.intermediateSteps = 5;
		oneClass2.setMaxOutliersInTraining(0.66);
		oneClass2.setKernelType(LibSVM.KERNELTYPE_LINEAR);
		//		oneClass2.setDegree(3);
		oneClass2.buildClassifier(instX);

		Random r = new Random();
		boolean jittering = true;

		XYSeries series = new XYSeries("series");
		int count = 0;
		CountedSet<String> set1 = new CountedSet<>();
		CountedSet<String> set2 = new CountedSet<>();
		double v1[] = new double[instX.size()];
		double v2[] = new double[instX.size()];
		for (Instance i : instX)
		{
			double x = oneClass1.pInside(i);
			double y = oneClass2.pInside(i);
			set1.add(StringUtil.formatDouble(x));
			set2.add(StringUtil.formatDouble(y));
			v1[count] = x;
			v2[count] = y;
			if (jittering)
			{
				x += r.nextDouble() * 0.05 * (r.nextBoolean() ? -1 : 1);
				y += r.nextDouble() * 0.05 * (r.nextBoolean() ? -1 : 1);
			}

			//series.add(lofOut.get(count).value(lofOut.numAttributes() - 1), oneClass.pInside(i));
			series.add(x, y);
			count++;
		}
		String s1 = Arrays.toString(oneClass1.getOptions());
		String s2 = Arrays.toString(oneClass2.getOptions());

		System.out.println("\n" + s1 + "\n" + set1);
		System.out.println("\n" + s2 + "\n" + set2);

		HistogramPanel p = new HistogramPanel("title", null, "xAxis", "yAxisLabel",
				Arrays.asList(s1, s2), Arrays.asList(v1, v2), 40);
		JFreeChart c = p.getChart();
		c.getPlot().setBackgroundPaint(Color.WHITE);
		c.getXYPlot().getRenderer().setSeriesPaint(0, new Color(255, 0, 0, 50));
		c.getXYPlot().getRenderer().setSeriesPaint(1, new Color(0, 0, 255, 50));
		c.setBorderPaint(Color.WHITE);
		SwingUtil.showInFrame(p, "hist", false);

		XYSeriesCollection xydataset = new XYSeriesCollection();
		xydataset.addSeries(series);
		JFreeChart xylineChart = ChartFactory.createScatterPlot("chartTitle", s1, s2, xydataset,
				PlotOrientation.VERTICAL, true, true, false);
		((XYLineAndShapeRenderer) xylineChart.getXYPlot().getRenderer()).setBaseShapesFilled(false);
		SwingUtil.showInFrame(new ChartPanel(xylineChart), "scatter", false);

		ScreenUtil.centerWindowsOnScreen();
		SwingUtil.waitWhileWindowsVisible();
		System.exit(0);
		// TODO
		// 1. test p-inside ranges with more then 0 intermediate steps
		// 2. plot this in a 2d plot
		// 3. recompute app-domain

	}

}
