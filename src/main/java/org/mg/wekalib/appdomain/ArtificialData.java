package org.mg.wekalib.appdomain;

import java.awt.Color;
import java.awt.Dimension;
import java.awt.Paint;
import java.awt.Shape;
import java.awt.geom.Ellipse2D;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import org.apache.commons.math3.distribution.NormalDistribution;
import org.apache.commons.math3.distribution.UniformRealDistribution;
import org.apache.commons.math3.random.RandomGenerator;
import org.apache.commons.math3.random.RandomGeneratorFactory;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.NumberAxis;
import org.jfree.chart.axis.NumberTickUnit;
import org.jfree.chart.plot.DatasetRenderingOrder;
import org.jfree.chart.plot.ValueMarker;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.renderer.xy.StandardXYItemRenderer;
import org.jfree.chart.renderer.xy.XYItemRenderer;
import org.jfree.chart.renderer.xy.XYLineAndShapeRenderer;
import org.jfree.data.function.Function2D;
import org.jfree.data.general.DatasetUtilities;
import org.jfree.data.xy.DefaultXYDataset;
import org.jfree.data.xy.XYDataset;
import org.jfree.data.xy.XYSeries;
import org.jfree.util.ShapeUtilities;
import org.mg.javalib.freechart.FreeChartUtil;
import org.mg.javalib.gui.property.ColorGradient;
import org.mg.javalib.util.ColorUtil;
import org.mg.javalib.util.ListUtil;
import org.mg.javalib.util.SwingUtil;

import weka.classifiers.Classifier;
import weka.classifiers.trees.RandomForest;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.EuclideanDistance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.neighboursearch.LinearNNSearch;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.LOF;

public class ArtificialData
{
	public double function(double x)
	{
		return Math.exp(x);
	}

	NormalDistribution noiseDistX;
	NormalDistribution noiseDistY;

	public boolean isTrue(double x, double y, boolean noise)
	{
		if (!noise)
			return y < function(x);
		else
		{
			if (noiseDistX == null)
			{
				Random r = new Random(13);
				RandomGenerator rg = RandomGeneratorFactory.createRandomGenerator(r);
				noiseDistX = new NormalDistribution(rg, 0, 0.05);
				noiseDistY = new NormalDistribution(rg, 0, 0.05);
			}
			double xOffset = noiseDistX.sample();
			double yOffset = noiseDistY.sample();
			return isTrue(x + xOffset, y + yOffset, false);
		}
	}

	Instances trainingData;
	Classifier clazzy;

	public void train(List<double[]> trueValues, List<double[]> falseValues)
	{
		Attribute x = new Attribute("x");
		Attribute y = new Attribute("y");
		Attribute clazz = new Attribute("clazz", ListUtil.createList("false", "true"));
		trainingData = new Instances("data",
				(ArrayList<Attribute>) ListUtil.createList(x, y, clazz), 0);
		trainingData.setClassIndex(2);
		for (double v[] : trueValues)
			trainingData.add(new DenseInstance(1.0, new double[] { v[0], v[1], 1.0 }));
		for (double v[] : falseValues)
			trainingData.add(new DenseInstance(1.0, new double[] { v[0], v[1], 0.0 }));
		System.out.println("instances " + trainingData.numInstances());
		try
		{
			clazzy.buildClassifier(trainingData);
		}
		catch (Exception e)
		{
			throw new RuntimeException(e);
		}
	}

	public boolean classify(double x, double y)
	{
		return classify(x, y, false);
	}

	public double predict(double x, double y)
	{
		DenseInstance inst = new DenseInstance(1.0, new double[] { x, y, Double.NaN });
		inst.setDataset(trainingData);
		try
		{
			double p = clazzy.distributionForInstance(inst)[1];
			System.out.println(p + " for " + x + "/" + y);
			return p;
		}
		catch (Exception e)
		{
			throw new RuntimeException(e);
		}
	}

	private LOF lof;

	public double lof(double x, double y)
	{
		if (lof == null)
		{
			try
			{
				lof = new LOF();
				lof.setInputFormat(trainingData);
				lof.setMinPointsLowerBound(2 + "");
				lof.setMinPointsUpperBound(4 + "");
				LinearNNSearch search = new LinearNNSearch();
				search.setDistanceFunction(new EuclideanDistance());
				lof.setNNSearch(search);
				lof.setNumExecutionSlots(1 + "");
				Filter.useFilter(trainingData, lof);
			}
			catch (Exception e)
			{
				throw new RuntimeException(e);
			}
		}

		//		ArrayList<Attribute> attributes = new ArrayList<>();
		//		for (int i = 0; i < trainingData.numAttributes(); i++)
		//			attributes.add(trainingData.attribute(i));
		//		Instances insts = new Instances("new", attributes, 1);
		//insts.add(inst);

		Instance inst = new DenseInstance(1.0, new double[] { x, y, Double.NaN });
		inst.setDataset(trainingData);

		//		System.out.println(trainingData.numInstances());
		//		System.out.println(insts.numInstances());

		try
		{
			lof.input(inst);
			Instance scored = lof.output();
			return scored.value(scored.numAttributes() - 1);
		}
		catch (Exception e)
		{
			throw new RuntimeException(e);
		}
	}

	public boolean classify(double x, double y, boolean verbose)
	{
		DenseInstance inst = new DenseInstance(1.0, new double[] { x, y, Double.NaN });
		inst.setDataset(trainingData);
		try
		{
			if (verbose)
			{
				System.out.println("\nclassify " + x + "/" + y);
				double predicted = clazzy.classifyInstance(inst);
				System.out.println("prediction " + (predicted == 1.0));
				System.out.println(
						"probability " + (clazzy.distributionForInstance(inst)[(int) predicted]));
			}

			return clazzy.classifyInstance(inst) == 1.0;
		}
		catch (Exception e)
		{
			throw new RuntimeException(e);
		}
	}

	enum Draw
	{
		OnlyTrain, UniverseUnpredicted, UniversePredicted, UniversePredictedBoundary, GroundTruth,
		LOF;

		public boolean addUniverse()
		{
			return this != Draw.OnlyTrain;
		}

		public boolean drawFunction()
		{
			return this == Draw.GroundTruth;
		}

		public boolean showUniversePrediction()
		{
			return this == Draw.UniversePredicted || this == Draw.UniversePredictedBoundary
					|| this == Draw.LOF;
		}

		public boolean drawBoundary()
		{
			return this == Draw.UniversePredictedBoundary;
		}

		public boolean showUniverseClass()
		{
			return this == GroundTruth;
		}
	}

	public ArtificialData(Classifier clazzy, boolean noise, long seed)
	{
		this.clazzy = clazzy;

		Random r = new Random(seed);
		RandomGenerator rg = RandomGeneratorFactory.createRandomGenerator(r);

		double xMin = -3.5;
		double xMax = 3.5;
		double yMin = -1.0;
		double yMax = 6.0;

		XYSeries seriesTrain = new XYSeries("train");
		List<double[]> trueTrain = new ArrayList<>();
		List<double[]> falseTrain = new ArrayList<>();
		NormalDistribution xNorm = new NormalDistribution(rg, 1.33, 0.5);
		NormalDistribution yNorm = new NormalDistribution(rg, function(1.33), 0.5);
		for (int i = 0; i < 200; i++)
		{
			double x = xNorm.sample();
			double y = yNorm.sample();
			if (isTrue(x, y, noise))
				trueTrain.add(new double[] { x, y });
			else
				falseTrain.add(new double[] { x, y });
		}
		//		xNorm = new NormalDistribution(rg, 0.7, 0.5);
		//		yNorm = new NormalDistribution(rg, function(0.3), 0.5);
		//		for (int i = 0; i < 200; i++)
		//		{
		//			double x = xNorm.sample();
		//			double y = yNorm.sample();
		//			if (isTrue(x, y, noise))
		//				trueTrain.add(new double[] { x, y });
		//			else
		//				falseTrain.add(new double[] { x, y });
		//		}

		for (double[] v : trueTrain)
			seriesTrain.add(v[0], v[1]);
		for (double[] v : falseTrain)
			seriesTrain.add(v[0], v[1]);
		train(trueTrain, falseTrain);

		XYSeries seriesUniverse = new XYSeries("universe");
		UniformRealDistribution xDist = new UniformRealDistribution(rg, xMin, xMax);
		UniformRealDistribution yDist = new UniformRealDistribution(rg, yMin, yMax);
		for (int i = 0; i < 200; i++)
		{
			double x = xDist.sample();
			double y = yDist.sample();
			seriesUniverse.add(x, y);
		}

		XYSeries seriesUniverseBoundary = new XYSeries("universe-boundary");
		long i = 0;
		while (seriesUniverseBoundary.getItemCount() < 100)
		{
			double x = xDist.sample();
			double y = yDist.sample();
			double p = predict(x, y);
			if (p > 0.4 && p < 0.6)
				seriesUniverseBoundary.add(x, y);
			i++;
			if (i % 10000000 == 0)
				System.out.println("sampling " + i);
		}

		//for (final Draw draw : ArrayUtil.reverse(Draw.values()))
		final Draw draw = Draw.UniversePredicted;
		{
			//			if (draw == Draw.UniversePredicted)
			//				continue;

			DefaultXYDataset d = new DefaultXYDataset();
			d.addSeries(seriesTrain.getKey(), seriesTrain.toArray());
			if (draw.addUniverse())
				d.addSeries(seriesUniverse.getKey(), seriesUniverse.toArray());
			if (draw.drawBoundary())
			{
				d.addSeries(seriesUniverseBoundary.getKey(), seriesUniverseBoundary.toArray());
			}

			JFreeChart f = ChartFactory.createScatterPlot(null, "x", "y", d);
			f.removeLegend();

			final XYPlot plot = f.getXYPlot();
			plot.setBackgroundPaint(Color.WHITE);
			ChartPanel cp = new ChartPanel(f);

			XYLineAndShapeRenderer renderer = new XYLineAndShapeRenderer()
			{
				private static final long serialVersionUID = 1L;

				//ColorGradient gradient = new ColorGradient(Color.RED,  Color.BLUE);

				@Override
				public Paint getItemPaint(int series, int column)
				{
					double x = plot.getDataset(0).getX(series, column).doubleValue();
					double y = plot.getDataset(0).getY(series, column).doubleValue();
					if (series == 0)
					{
						if (isTrue(x, y, true))
							//if (classify(x, y))
							return Color.RED;
						else
							return Color.BLUE;
					}
					else // universe or boundary
					{
						Color c;

						if (draw.showUniversePrediction())
						{
							if (draw == Draw.LOF)
							{
								double lof = lof(x, y);
								if (lof < 1.75)
									c = Color.GREEN;
								else if (lof < 2.25)
									c = Color.GRAY;
								else
									c = Color.ORANGE;
							}
							else
							{
								c = ColorGradient.get2ColorGradient(predict(x, y), Color.RED,
										Color.BLUE);
								//								c = new ColorGradient(Color.GREEN, Color.GRAY, Color.MAG)
								//										.getColor((predict(x, y) - 0.5) * 2);
							}
							//							c = gradient.getColor(predict(x, y));
						}
						else if (draw.showUniverseClass())
						{
							if (isTrue(x, y, true))
								c = Color.RED;
							else
								c = Color.BLUE;
						}
						else
						{
							c = Color.GRAY;
						}

						return ColorUtil.transparent(c, 150);
					}
					//					if (row == 0)
					//					{
					//						return Color.RED;
					//						
					//						
					//					}
					//					else if (row == 1)
					//					{
					//						return Color.BLUE;
					//					}
					//					else if (row == 2)
					//					{
					//						//plot.getDataset(0).getX(row, column);
					//						return Color.RED;
					//					}
					//					else if (row == 3)
					//					{
					//						return Color.BLUE;
					//					}
					//					else
					//						throw new IllegalArgumentException();
				}
			};
			renderer.setBaseLinesVisible(false);

			float size = 4f;
			Shape circle = new Ellipse2D.Double(-size, -size, size * 2, size * 2);
			Shape shape = ShapeUtilities.createDiamond(size);
			renderer.setSeriesShape(0, shape);
			renderer.setSeriesShape(1, circle);
			renderer.setSeriesShape(2, circle);
			renderer.setSeriesShape(3, circle);
			//			renderer.setSeriesShape(2, shape);
			//			renderer.setSeriesShape(3, shape);
			renderer.setSeriesShapesFilled(0, true);
			renderer.setSeriesShapesFilled(1, true);
			renderer.setSeriesShapesFilled(2, true);
			renderer.setSeriesPaint(0, Color.GRAY);
			renderer.setSeriesPaint(1, Color.GRAY);
			renderer.setSeriesPaint(2, Color.GRAY);
			//			renderer.setSeriesPaint(0, Color.RED);
			//			renderer.setSeriesPaint(1, Color.BLUE);
			//				renderer.setSeriesShapesFilled(2, false);
			//				renderer.setSeriesShapesFilled(3, false);
			//					renderer.setSeriesPaint(2, Color.GRAY);
			if (draw.drawFunction())
			{
				//					renderer.setSeriesPaint(2, Color.RED);
				//					renderer.setSeriesPaint(3, Color.BLUE);

				{
					Function2D func = new Function2D()
					{
						@Override
						public double getValue(double x)
						{
							return function(x);
						}
					};
					XYDataset result = DatasetUtilities.sampleFunction2D(func, xMin, xMax, 300,
							"exp");
					plot.setDataset(1, result);
					final XYItemRenderer renderer2 = new StandardXYItemRenderer();
					plot.setRenderer(1, renderer2);
					//			final ValueAxis rangeAxis2 = new NumberAxis("exp");
					//			plot.setRangeAxis(1, rangeAxis2);
					//			plot.mapDatasetToRangeAxis(1, 1);
					renderer2.setSeriesPaint(0, Color.BLACK);
					plot.setDatasetRenderingOrder(DatasetRenderingOrder.FORWARD);
					//			rangeAxis2.setLabelFont(plot.getRangeAxis().getLabelFont());
					//			rangeAxis2.setTickLabelFont(plot.getRangeAxis().getTickLabelFont());

					ValueMarker marker = new ValueMarker(0);
					marker.setPaint(Color.black);
					plot.addDomainMarker(marker);
					plot.addRangeMarker(marker);
				}
			}
			plot.setRenderer(renderer);

			((NumberAxis) plot.getRangeAxis()).setTickUnit(new NumberTickUnit(1.0));
			((NumberAxis) plot.getDomainAxis()).setTickUnit(new NumberTickUnit(1.0));

			//			if (draw == Draw.OnlyTrain)
			//			{
			//				plot.getRangeAxis().setRange(2.5, 5.5);
			//				plot.getDomainAxis().setRange(0, 3);
			//			}
			//			else
			//			{
			plot.getRangeAxis().setRange(yMin, yMax);
			plot.getDomainAxis().setRange(xMin, xMax);
			//			}

			cp.setMaximumDrawWidth(100000);
			cp.setMaximumDrawHeight(100000);
			cp.setPreferredSize(new Dimension(800, 800));

			String name = draw.toString();
			if (draw == Draw.UniversePredictedBoundary)
				name += " " + clazzy.getClass().getSimpleName() + " " + seed;
			FreeChartUtil.toPNGFile(
					"/home/martin/documents/app-domain/imgs/" + name.replaceAll(" ", "_") + ".png",
					cp, cp.getPreferredSize());

			SwingUtil.showInFrame(cp, name, false);
		}
	}

	public static void main(String args[])
	{
		long seed = new Random().nextLong();
		System.err.println(seed);
		//new ArtificialData(new RandomForest(), true, seed);

		RandomForest one = new RandomForest();
		//OneClassSVM one = new OneClassSVM();
		//Classifier one = new DistanceBasedADClassifier(new PCAEuclideanADModel(3, true, 0.95), 1);

		new ArtificialData(one, true, seed);

		//new ArtificialData(new LOF(), true, 2);
		//		new ArtificialData(new RandomForest(), true, 17);
		//		new ArtificialData(new RandomForest(), true, 2);

		//		ScreenUtil.centerWindowsOnScreen();
		SwingUtil.waitWhileWindowsVisible();
		System.exit(0);
	}
}
