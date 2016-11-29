package org.mg.wekalib.appdomain;

import java.awt.Color;
import java.awt.Dimension;
import java.io.FileReader;
import java.util.Random;

import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.XYPlot;
import org.jfree.data.xy.DefaultXYDataset;
import org.jfree.data.xy.XYSeries;
import org.mg.javalib.util.ListUtil;
import org.mg.javalib.util.SwingUtil;
import org.mg.wekalib.data.InstanceUtil;
import org.mg.wekalib.distance.EuclideanPCADistance;
import org.mg.wekalib.eval2.data.WekaAntiStratifiedSplitter;

import weka.attributeSelection.PrincipalComponents;
import weka.core.Attribute;
import weka.core.EuclideanDistance;
import weka.core.Instances;

public class AppDomainTest
{
	public static void testTanimoto() throws Exception
	{
		long seed = new Random().nextLong();
		//seed = -6455245800635986416L;
		System.out.println("seed " + seed);
		Random r = new Random(seed);

		String data = "vote";
		Instances inst = new Instances(new FileReader(
				System.getProperty("user.home") + "/data/weka/nominal/" + data + ".arff"));
		inst.setClassIndex(inst.numAttributes() - 1);
		inst.randomize(r);

		//		Instances split[] = SplitDatasets.antiStratifiedSplit(inst, 0.5, new TanimotoDistance(),
		//				r.nextLong());
		Instances split[] = WekaAntiStratifiedSplitter.split(inst, 0.66);
		Instances train = split[0];
		Instances test = split[1];

		System.out.println("train test split " + train.size() + "/" + test.size());

		DistanceBasedADModel ad = new TanimotoCentroidADModel(0.95, 0.1);
		ad.build(train);

		int inside = 0;
		for (int i = 0; i < test.size(); i++)
		{
			if (ad.isInsideAppdomain(test.get(i)))
			{
				System.out.println("inside");
				inside++;
			}
			else
			{
				System.out.println("outside");
				//				SwingUtil.showInFrame(ad.getPlot(test.get(i)));
			}
		}
		System.out.println(
				inside + "/" + test.size() + " inside (" + (inside / (double) test.size()) + ")");

		//		SwingUtil.waitWhileWindowsVisible();
		System.exit(0);
	}

	public static void main(String[] args) throws Exception
	{
		testTanimoto();
		System.exit(0);

		long seed = new Random().nextLong();
		//seed = -6455245800635986416L;
		System.out.println("seed " + seed);
		Random r = new Random(seed);

		//String data = "vote";
		//String data = "sonar";
		String data = "iris-easy";
		//String data = "segment";
		Instances inst = new Instances(new FileReader(
				System.getProperty("user.home") + "/data/weka/nominal/" + data + ".arff"));
		inst.setClassIndex(inst.numAttributes() - 1);
		int maxNumAttributes = 2;
		while (inst.numAttributes() > maxNumAttributes)
			inst = InstanceUtil.stripAttributes(inst,
					ListUtil.create(Attribute.class, inst.attribute(2)));//r.nextInt(inst.numAttributes()))));
		//		FileUtil.writeStringToFile("/home/martin/data/weka/nominal/iris.easy.arff",
		//				inst.toString());
		//		System.exit(1);

		//		inst = InstanceUtil.stripAttributes(inst, ListUtil.create(Attribute.class,
		//				inst.attribute(2), inst.attribute(3), inst.attribute(4)));

		//inst.setClassIndex(inst.numAttributes() - 1);
		//		System.out.println(inst);

		inst.randomize(r);

		Instances split[] = WekaAntiStratifiedSplitter.antiStratifiedSplit(inst, 0.5,
				//
				//		new TanimotoDistance());
				new EuclideanPCADistance(), r.nextLong());
		Instances train = split[0];
		Instances test = split[1];

		System.out.println("train test split " + train.size() + "/" + test.size());

		//Instances train = new Instances(new FileReader("/tmp/train.arff"));
		//Instances test = new Instances(new FileReader("/tmp/test.arff"));

		//		//DistanceBasedADModel ad = new TanimotoADModel(3, true, 0.90);//, 0, Integer.MAX_VALUE);
		//		DistanceBasedADModel ad = new PCAEuclideanADModel(1, true, 0.8);
		//		ad.build(train);
		//
		//		//		System.out.println("0.1 " + ad.getCumulativeProbability(0.1));
		//		//		System.out.println("0.2 " + ad.getCumulativeProbability(0.2));
		//		//		System.out.println("0.3 " + ad.getCumulativeProbability(0.3));
		//
		//		int inside = 0;
		//		//		List<Double> probInside = new ArrayList<>();
		//		for (int i = 0; i < test.size(); i++)
		//		{
		//
		//			if (ad.isInsideAppdomain(test.get(i)))
		//			{
		//				//				System.out.println("inside");
		//				inside++;
		//			}
		//			else
		//			{
		//				//				System.out.println("outside");
		//
		//				//				SwingUtil.showInFrame(ad.getPlot(test.get(i)));
		//			}
		//
		//			//			System.out.println(StringUtil.formatDouble(ad.computeDistanceToTraining(test.get(i)))
		//			//					+ " " + StringUtil.formatDouble(ad
		//			//							.getCumulativeProbability(ad.computeDistanceToTraining(test.get(i)))));
		//			//			System.out.println("\n");
		//
		//			//			probInside.add(ad.insideProbability(inst.get(i)));
		//		}
		//		System.out.println(
		//				inside + "/" + test.size() + " inside (" + (inside / (double) test.size()) + ")");

		//		SwingUtil.showInFrame(ad.getPlot(ListUtil.toArray(test)), "plot", false);

		pcaPlot(inst, train, test, null);

		//		System.out.println(DoubleArraySummary.create(probInside));

		SwingUtil.waitWhileWindowsVisible();

		System.exit(0);
	}

	public static void pcaPlot(Instances all, Instances train, Instances test,
			AbstractDistanceBasedADModel ad)
	{
		try
		{
			PrincipalComponents pca = new PrincipalComponents();
			pca.setCenterData(false);
			pca.setVarianceCovered(0.95);
			pca.buildEvaluator(all);

			System.out.println("num attributes " + train.numAttributes());
			System.out.println("num attributes " + test.numAttributes());

			DefaultXYDataset d = new DefaultXYDataset();
			XYSeries series;

			Instances ttrain = pca.transformedData(train);
			System.out.println("num attributes ttrain " + ttrain.numAttributes());
			series = new XYSeries("train");
			for (int i = 0; i < train.size(); i++)
				series.add(ttrain.get(i).value(0), ttrain.get(i).value(1));
			d.addSeries(series.getKey(), series.toArray());

			Instances ttest = pca.transformedData(test);
			System.out.println("num attributes ttest " + ttest.numAttributes());
			series = new XYSeries("test");
			for (int i = 0; i < ttest.size(); i++)
				series.add(ttest.get(i).value(0), ttest.get(i).value(1));
			d.addSeries(series.getKey(), series.toArray());

			EuclideanDistance dist = new EuclideanDistance(pca.transformedData(all));
			DescriptiveStatistics stats = new DescriptiveStatistics();
			for (int i = 0; i < ttrain.numInstances(); i++)
				for (int j = 0; j < ttest.numInstances(); j++)
					stats.addValue(dist.distance(ttrain.get(i), ttest.get(j)));
			System.out.println("distance between test and train: " + stats.getMean() + " +- "
					+ stats.getStandardDeviation());

			//			{
			//				Instances ttest = pca.transformedData(test);
			//				XYSeries series = new XYSeries("test-outside");
			//				for (int i = 0; i < ttest.size(); i++)
			//					if (!ad.isInsideAppdomain(test.get(i)))
			//						series.add(ttest.get(i).value(0), ttest.get(i).value(1));
			//				d.addSeries(series.getKey(), series.toArray());
			//			}
			//
			//			{
			//				Instances ttest = pca.transformedData(test);
			//				XYSeries series = new XYSeries("test-inside");
			//				for (int i = 0; i < ttest.size(); i++)
			//					if (ad.isInsideAppdomain(test.get(i)))
			//						series.add(ttest.get(i).value(0), ttest.get(i).value(1));
			//				d.addSeries(series.getKey(), series.toArray());
			//			}

			JFreeChart f = ChartFactory.createScatterPlot("PCA", "pc1", "pc2", d);

			XYPlot plot = f.getXYPlot();
			plot.setBackgroundPaint(Color.WHITE);
			//			XYLineAndShapeRenderer renderer = (XYLineAndShapeRenderer) plot.getRenderer();
			//			renderer.setBaseShapesFilled(false);

			ChartPanel cp = new ChartPanel(f);

			cp.setMaximumDrawWidth(100000);
			cp.setMaximumDrawHeight(100000);
			cp.setPreferredSize(new Dimension(600, 600));

			SwingUtil.showInFrame(cp);
		}
		catch (Exception e)
		{
			e.printStackTrace();
		}
	}
}
