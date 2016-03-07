package org.mg.wekalib.evaluation;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;

import org.apache.commons.lang3.ArrayUtils;
import org.apache.commons.math3.stat.correlation.PearsonsCorrelation;
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.apache.commons.math3.util.FastMath;
import org.asad.virtualscreening.enrichvs.EnrichmentAssessment;
import org.mg.javalib.util.ArrayUtil;
import org.mg.javalib.util.CountedSet;
import org.mg.javalib.util.DoubleArraySummary;

import weka.classifiers.Evaluation;
import weka.classifiers.evaluation.NominalPrediction;
import weka.classifiers.evaluation.Prediction;
import weka.classifiers.evaluation.ThresholdCurve;
import weka.core.Instances;

public class PredictionUtil
{
	public static Predictions concat(Predictions p1, Predictions p2)
	{
		if (p1.actual == null)
		{
			p1.actual = new double[0];
			p1.predicted = new double[0];
			p1.confidence = new double[0];
			p1.fold = new int[0];
			p1.origIndex = new int[0];
		}
		if (p2.actual == null)
		{
			p2.actual = new double[0];
			p2.predicted = new double[0];
			p2.confidence = new double[0];
			p2.fold = new int[0];
			p2.origIndex = new int[0];
		}
		Predictions p = new Predictions();
		p.fold = ArrayUtil.concat(p1.fold, p2.fold);
		p.actual = ArrayUtil.concat(p1.actual, p2.actual);
		p.predicted = ArrayUtil.concat(p1.predicted, p2.predicted);
		p.confidence = ArrayUtil.concat(p1.confidence, p2.confidence);
		p.origIndex = ArrayUtil.concat(p1.origIndex, p2.origIndex);
		return p;
	}

	public static Predictions clone(Predictions pred)
	{
		Predictions p = new Predictions();
		p.actual = Arrays.copyOf(pred.actual, pred.actual.length);
		p.predicted = Arrays.copyOf(pred.predicted, pred.actual.length);
		p.confidence = Arrays.copyOf(pred.confidence, pred.actual.length);
		p.fold = Arrays.copyOf(pred.fold, pred.actual.length);
		p.origIndex = Arrays.copyOf(pred.origIndex, pred.actual.length);
		return p;
	}

	public static Predictions add(Predictions pred, int fold, double[] act, double[] prd,
			double conf[], int oIdx[])
	{
		Predictions p = new Predictions();

		int f[] = new int[act.length];
		for (int i = 0; i < f.length; i++)
			f[i] = fold;

		if (pred == null)
		{
			p.fold = f;
			p.actual = act;
			p.predicted = prd;
			p.origIndex = oIdx;
			p.confidence = conf;
		}
		else
		{
			p.fold = ArrayUtil.concat(pred.fold, f);
			p.actual = ArrayUtil.concat(pred.actual, act);
			p.predicted = ArrayUtil.concat(pred.predicted, prd);
			p.confidence = ArrayUtil.concat(pred.confidence, conf);
			p.origIndex = ArrayUtil.concat(pred.origIndex, oIdx);
		}

		return p;
	}

	public static void add(Predictions p, Predictions source, int sourceIdx)
	{
		add(p, source.actual[sourceIdx], source.predicted[sourceIdx], source.confidence[sourceIdx],
				source.fold[sourceIdx], source.origIndex[sourceIdx]);
	}

	public static void add(Predictions p, double actual, double predicted, double confidence,
			int fold, int origIdx)
	{
		if (p.actual == null)
		{
			p.actual = new double[0];
			p.predicted = new double[0];
			p.confidence = new double[0];
			p.fold = new int[0];
			p.origIndex = new int[0];
		}
		p.actual = ArrayUtil.push(p.actual, actual);
		p.predicted = ArrayUtil.push(p.predicted, predicted);
		p.confidence = ArrayUtil.push(p.confidence, confidence);
		p.fold = ArrayUtil.push(p.fold, fold);
		p.origIndex = ArrayUtil.push(p.origIndex, origIdx);

	}

	public static List<Predictions> perFold(Predictions preds)
	{
		List<Predictions> pPerFold = new ArrayList<>();

		int minMax[] = ArrayUtil.getMinMax(preds.fold);
		for (int fold = minMax[0]; fold <= minMax[1]; fold++)
		{
			Predictions p = new Predictions();
			for (int j = 0; j < preds.actual.length; j++)
				if (preds.fold[j] == fold)
					add(p, preds, j);
			pPerFold.add(p);
		}

		return pPerFold;
	}

	public static double[] pearsonPerFold(Predictions preds)
	{
		List<Predictions> pPerFold = perFold(preds);
		double[] d = new double[pPerFold.size()];
		for (int i = 0; i < d.length; i++)
			d[i] = pearson(pPerFold.get(i));
		return d;
	}

	//	public static List<Predictions> perDrugCombi(Predictions pred, List<String> drugCombination)
	//	{
	//		HashMap<String, List<Integer>> drugCombinationToIdx = new LinkedHashMap<>();
	//		for (String drugC : new LinkedHashSet<String>(drugCombination))
	//			drugCombinationToIdx.put(drugC, new ArrayList<Integer>());
	//		int i = 0;
	//		for (String drugC : drugCombination)
	//			drugCombinationToIdx.get(drugC).add(i++);
	//
	//		int[] origToPredIdx = new int[drugCombination.size()];
	//		for (int j = 0; j < pred.origIndex.length; j++)
	//			origToPredIdx[pred.origIndex[j]] = j;
	//
	//		List<Predictions> pPerDrug = new ArrayList<>();
	//
	//		for (String drugC : drugCombinationToIdx.keySet())
	//		{
	//			List<Integer> indices = drugCombinationToIdx.get(drugC);
	//			Predictions p = new Predictions();
	//			for (Integer idx : indices)
	//				add(p, pred, origToPredIdx[idx]);
	//			pPerDrug.add(p);
	//		}
	//
	//		return pPerDrug;
	//	}

	private static class InfoPredictions extends Predictions
	{
		public String info[];
	}

	private static InfoPredictions attachInfo(Predictions pred, List<String> info)
	{
		InfoPredictions p = new InfoPredictions();
		p.actual = Arrays.copyOf(pred.actual, pred.actual.length);
		p.predicted = Arrays.copyOf(pred.predicted, pred.actual.length);
		p.confidence = Arrays.copyOf(pred.confidence, pred.actual.length);
		p.fold = Arrays.copyOf(pred.fold, pred.actual.length);
		p.origIndex = Arrays.copyOf(pred.origIndex, pred.actual.length);

		p.info = new String[pred.actual.length];
		for (int i = 0; i < pred.actual.length; i++)
			p.info[i] = info.get(p.origIndex[i]);

		return p;
	}

	private static HashMap<String, Predictions> split(Predictions pred, List<String> info)
	{
		HashMap<String, Predictions> m = new LinkedHashMap<>();
		InfoPredictions predI = attachInfo(pred, info);
		for (int i = 0; i < predI.actual.length; i++)
		{
			String inf = predI.info[i];
			if (!m.containsKey(inf))
				m.put(inf, new Predictions());
			Predictions p = m.get(inf);
			PredictionUtil.add(p, pred, i);
		}
		return m;
	}

	public static List<Predictions> perDrugCombi(Predictions pred, List<String> drugCombination,
			double percentage)
	{
		List<Predictions> split = new ArrayList<>(split(pred, drugCombination).values());

		if (percentage < 1.0)
		{
			DescriptiveStatistics stats = new DescriptiveStatistics();
			for (Predictions p : split)
			{
				System.out.println(p.actual.length);
				//				double mean = new DescriptiveStatistics(p.confidence).getMean();
				double mean = new DescriptiveStatistics(p.predicted).getVariance();
				stats.addValue(mean);
			}
			System.out.print(split.size() + " -> ");
			double minMean = stats.getPercentile((1.0 - percentage) * 100);
			for (Predictions p : new ArrayList<>(split))
			{
				//double mean = new DescriptiveStatistics(p.confidence).getMean();
				double mean = new DescriptiveStatistics(p.predicted).getVariance();
				if (mean < minMean)
					split.remove(p);
			}
			System.out.println(split.size());
		}

		return split;

		//		List<Predictions> split = new ArrayList<>(split(pred, drugCombination).values());
		//
		//		if (percentage < 1.0)
		//		{
		//
		//			DescriptiveStatistics conf = new DescriptiveStatistics();
		//			for (Predictions p : split)
		//				conf.addValue(new DescriptiveStatistics(p.confidence).getMean());
		//			double confDelta = conf.getMax() - conf.getMin();
		//			double confMin = conf.getMin();
		//
		//			DescriptiveStatistics var = new DescriptiveStatistics();
		//			for (Predictions p : split)
		//				var.addValue(new DescriptiveStatistics(p.predicted).getVariance());
		//			double varDelta = var.getMax() - var.getMin();
		//			double varMin = var.getMin();
		//
		//			DescriptiveStatistics stats = new DescriptiveStatistics();
		//			//			List<Integer> sizes = new ArrayList<>();
		//			for (Predictions p : split)
		//			{
		//				//				sizes.add(p.actual.length);
		//				double c = 0;//(new DescriptiveStatistics(p.confidence).getMean() - confMin) / confDelta;
		//				//double v = (new DescriptiveStatistics(p.predicted).getVariance() - varMin) / varDelta;
		//				double v = new DescriptiveStatistics(p.predicted).getVariance();
		//				//				System.out.println(v);
		//				//double mean = new DescriptiveStatistics(p.predicted).getVariance();
		//				stats.addValue(v);// + v);
		//			}
		//
		//			//			System.out.println(DoubleArraySummary.create(sizes).toStringSummary());
		//
		//			System.out.print(split.size() + " -> ");
		//			double minVal = stats.getPercentile((1.0 - percentage) * 100);
		//			for (Predictions p : new ArrayList<>(split))
		//			{
		//				double c = 0;//(new DescriptiveStatistics(p.confidence).getMean() - confMin) / confDelta;
		//				//double v = (new DescriptiveStatistics(p.predicted).getVariance() - varMin) / varDelta;
		//				double v = new DescriptiveStatistics(p.predicted).getVariance();
		//				//double mean = new DescriptiveStatistics(p.predicted).getVariance();
		//				double val = v; //c + v;
		//
		//				if (val < minVal)
		//					split.remove(p);
		//			}
		//			System.out.println(split.size());
		//		}
		//
		//		return split;		

		//		HashMap<String, List<Integer>> drugCombinationToIdx = new LinkedHashMap<>();
		//		for (String drugC : new LinkedHashSet<String>(drugCombination))
		//			drugCombinationToIdx.put(drugC, new ArrayList<Integer>());
		//		int i = 0;
		//		for (String drugC : drugCombination)
		//			drugCombinationToIdx.get(drugC).add(i++);
		//
		//		int[] origToPredIdx = new int[drugCombination.size()];
		//		for (int j = 0; j < pred.origIndex.length; j++)
		//			origToPredIdx[pred.origIndex[j]] = j;
		//
		//		double quota = -Double.MAX_VALUE;
		//		HashMap<String, Double> drugCombinationConf = new LinkedHashMap<>();
		//		if (percentage < 1.0)
		//		{
		//			DescriptiveStatistics stats = new DescriptiveStatistics();
		//			for (String drugC : drugCombinationToIdx.keySet())
		//			{
		//				List<Integer> indices = drugCombinationToIdx.get(drugC);
		//				double sum = 0;
		//				for (Integer idx : indices)
		//					sum += pred.confidence[origToPredIdx[idx]];
		//				double conf = sum / (double) indices.size();
		//				stats.addValue(conf);
		//				drugCombinationConf.put(drugC, conf);
		//			}
		//			quota = stats.getPercentile((1.0 - percentage) * 100);
		//		}
		//
		//		List<Predictions> pPerDrug = new ArrayList<>();
		//		for (String drugC : drugCombinationToIdx.keySet())
		//		{
		//			if (percentage < 1 && drugCombinationConf.get(drugC) < quota)
		//				continue;
		//			List<Integer> indices = drugCombinationToIdx.get(drugC);
		//			Predictions p = new Predictions();
		//			for (Integer idx : indices)
		//				add(p, pred, origToPredIdx[idx]);
		//			pPerDrug.add(p);
		//		}
		//
		//		return pPerDrug;
	}

	//	public static List<Predictions> perDrugCombi(Predictions pred, List<String> drugCombination, double percentage)
	//	{
	//		HashMap<String, List<Integer>> drugCombinationToIdx = new LinkedHashMap<>();
	//		for (String drugC : new LinkedHashSet<String>(drugCombination))
	//			drugCombinationToIdx.put(drugC, new ArrayList<Integer>());
	//		int i = 0;
	//		for (String drugC : drugCombination)
	//			drugCombinationToIdx.get(drugC).add(i++);
	//
	//		int[] origToPredIdx = new int[drugCombination.size()];
	//		for (int j = 0; j < pred.origIndex.length; j++)
	//			origToPredIdx[pred.origIndex[j]] = j;
	//
	//		double quota = -Double.MAX_VALUE;
	//		HashMap<String, Double> drugCombinationConf = new LinkedHashMap<>();
	//		if (percentage < 1.0)
	//		{
	//			DescriptiveStatistics stats = new DescriptiveStatistics();
	//			for (String drugC : drugCombinationToIdx.keySet())
	//			{
	//				List<Integer> indices = drugCombinationToIdx.get(drugC);
	//				double sum = 0;
	//				for (Integer idx : indices)
	//					sum += pred.confidence[origToPredIdx[idx]];
	//				double conf = sum / (double) indices.size();
	//				stats.addValue(conf);
	//				drugCombinationConf.put(drugC, conf);
	//			}
	//			quota = stats.getPercentile((1.0 - percentage) * 100);
	//		}
	//
	//		List<Predictions> pPerDrug = new ArrayList<>();
	//		for (String drugC : drugCombinationToIdx.keySet())
	//		{
	//			if (percentage < 1 && drugCombinationConf.get(drugC) < quota)
	//				continue;
	//			List<Integer> indices = drugCombinationToIdx.get(drugC);
	//			Predictions p = new Predictions();
	//			for (Integer idx : indices)
	//				add(p, pred, origToPredIdx[idx]);
	//			pPerDrug.add(p);
	//		}
	//
	//		return pPerDrug;
	//	}

	private static double partialCorrelation(double u[], double v[], double w[])
	{
		PearsonsCorrelation c = new PearsonsCorrelation();
		double numerator = c.correlation(u, v) - c.correlation(u, w) * c.correlation(w, v);
		double denumerator = FastMath.sqrt(1 - Math.pow(c.correlation(u, w), 2))
				* FastMath.sqrt(1 - Math.pow(c.correlation(w, v), 2));
		return (numerator / denumerator);
	}

	private static HashMap<String, Double> getActualMedianForInfo(Predictions pred,
			List<String> info)
	{
		HashMap<String, Double> res = new LinkedHashMap<>();
		HashMap<String, Predictions> split = split(pred, info);
		for (String inf : split.keySet())
			res.put(inf, new DescriptiveStatistics(split.get(inf).actual).getPercentile(50));
		return res;
	}

	public static double globalPearson(Predictions pred, List<String> drugCombination,
			List<String> cellLine)
	{
		HashMap<String, Double> medianForDrugC = getActualMedianForInfo(pred, drugCombination);
		HashMap<String, Double> medianForCellL = getActualMedianForInfo(pred, cellLine);

		double x[] = pred.actual;
		double y[] = pred.predicted;

		double z0[] = new double[pred.actual.length];
		InfoPredictions predCellL = attachInfo(pred, cellLine);
		for (int i = 0; i < predCellL.actual.length; i++)
			z0[i] = medianForCellL.get(predCellL.info[i]);

		double z1[] = new double[pred.actual.length];
		InfoPredictions predDrugC = attachInfo(pred, drugCombination);
		for (int i = 0; i < predDrugC.actual.length; i++)
			z1[i] = medianForDrugC.get(predDrugC.info[i]);

		double numerator = partialCorrelation(x, y, z1)
				- partialCorrelation(x, z0, z1) * partialCorrelation(z0, y, z1);
		double denumerator = FastMath.sqrt(1 - Math.pow(partialCorrelation(x, z0, z1), 2))
				* FastMath.sqrt(1 - Math.pow(partialCorrelation(z0, y, z1), 2));

		return numerator / denumerator;
	}

	public static double pearsonPerDrugCombi(Predictions preds, List<String> drugCombination)
	{
		return pearsonPerDrugCombi(preds, drugCombination, 1.0);
	}

	public static double pearsonPerDrugCombi(Predictions preds, List<String> drugCombination,
			double percentage)
	{
		//List<String> uniqDrugs = new ArrayList<>(new LinkedHashSet<String>(drugCombination));

		List<Predictions> pPerDrug = perDrugCombi(preds, drugCombination, percentage);
		int pCount = 0;
		double pSum = 0;
		//		List<Double> res = new ArrayList<>();
		//		int idx = -1;
		for (Predictions p : pPerDrug)
		{
			//			idx++;
			double d = Double.NaN;
			if (p.actual.length > 1)
				d = pearson(p);
			//System.out.println(idx + " " + uniqDrugs.get(idx) + " " + p.actual.length + " " + d + " ");
			//			if (idx == 111)
			//			{
			//				System.out.println(ArrayUtil.toString(p.actual));
			//				System.out.println(ArrayUtil.toString(p.predicted));
			//				System.exit(0);
			//			}

			if (Double.isNaN(d))
				d = 0; //continue;
			//			res.add(d);
			pSum += d;
			pCount++;
		}
		//System.out.println(pCount);
		//System.out.println(pSum);
		//		System.out.println(pSum / (double) pCount);

		//		return ArrayUtil.toPrimitiveDoubleArray(res);
		return pSum / (double) pCount;
	}

	public static double[] pearsonPerFoldAndDrugCombi(Predictions preds,
			List<String> drugCombination, double percentage)
	{
		List<Predictions> pPerFold = perFold(preds);
		double[] d = new double[pPerFold.size()];
		for (int i = 0; i < d.length; i++)
			d[i] = pearsonPerDrugCombi(pPerFold.get(i), drugCombination, percentage);
		return d;
	}

	//	public static double[] pearsonPerFoldOld(Predictions preds)
	//	{
	//		List<Double> pPerFold = new ArrayList<>();
	//
	//		PearsonsCorrelation corr = new PearsonsCorrelation();
	//		int minMax[] = ArrayUtil.getMinMax(preds.fold);
	//		for (int i = minMax[0]; i <= minMax[1]; i++)
	//		{
	//			List<Double> a = new ArrayList<>();
	//			List<Double> p = new ArrayList<>();
	//			for (int j = 0; j < preds.actual.length; j++)
	//			{
	//				if (preds.fold[j] == i)
	//				{
	//					a.add(preds.actual[j]);
	//					p.add(preds.predicted[j]);
	//				}
	//			}
	//			pPerFold.add(corr.correlation(ArrayUtil.toPrimitiveDoubleArray(a), ArrayUtil.toPrimitiveDoubleArray(p)));
	//		}
	//
	//		return ArrayUtil.toPrimitiveDoubleArray(pPerFold);
	//	}

	public static double pearson(Predictions pred)
	{
		PearsonsCorrelation corr = new PearsonsCorrelation();
		return corr.correlation(pred.actual, pred.predicted);
	}

	public static double rmse(Predictions pred)
	{
		double ms = 0;
		for (int i = 0; i < pred.actual.length; i++)
		{
			double diff = pred.actual[i] - pred.predicted[i];
			ms += diff * diff;
		}
		ms /= (double) pred.actual.length;
		return Math.sqrt(ms);

	}

	//	public static double pearsonPerDrugCombiOld(Predictions pred, List<String> drugCombination)
	//	{
	//		PearsonsCorrelation corr = new PearsonsCorrelation();
	//
	//		HashMap<String, List<Integer>> drugCombinationToIdx = new HashMap<>();
	//		for (String drugC : new HashSet<String>(drugCombination))
	//			drugCombinationToIdx.put(drugC, new ArrayList<Integer>());
	//		int i = 0;
	//		for (String drugC : drugCombination)
	//			drugCombinationToIdx.get(drugC).add(i++);
	//
	//		int[] origToPredIdx = new int[pred.origIndex.length];
	//		for (int j = 0; j < pred.origIndex.length; j++)
	//			origToPredIdx[pred.origIndex[j]] = j;
	//
	//		double corrSum = 0;
	//		int corrCount = 0;
	//		for (String drugC : drugCombinationToIdx.keySet())
	//		{
	//			List<Integer> indices = drugCombinationToIdx.get(drugC);
	//
	//			double a[] = new double[indices.size()];
	//			double p[] = new double[indices.size()];
	//			i = 0;
	//			for (Integer idx : indices)
	//			{
	//				a[i] = pred.actual[origToPredIdx[idx]];
	//				p[i] = pred.predicted[origToPredIdx[idx]];
	//				i++;
	//			}
	//
	//			if (indices.size() <= 1)
	//				continue;
	//
	//			double v = corr.correlation(a, p);
	//
	//			if (Double.isNaN(v))
	//				continue;
	//			//			List<Integer> realOrigIdx = new ArrayList<>();
	//			//			for (Integer idx : indices)
	//			//				if (idx % 2 == 0)
	//			//					realOrigIdx.add(1 + 1 + (idx / 2));
	//			//			System.out.println(StringUtil.concatWhitespace(drugC, 25) + " " + indices.size() + " " + v + " "
	//			//					+ realOrigIdx);
	//			corrSum += v;
	//			corrCount++;
	//		}
	//		//			System.out.println(corrCount + " " + corrSum + " " + (corrSum / (double) corrCount));
	//		return corrSum / (double) corrCount;
	//	}

	public static Predictions intoHalf(Predictions pred)
	{
		Predictions p = new Predictions();
		p.actual = new double[pred.actual.length / 2];
		p.predicted = new double[pred.actual.length / 2];
		p.confidence = new double[pred.actual.length / 2];
		p.fold = new int[pred.actual.length / 2];
		p.origIndex = new int[pred.actual.length / 2];

		for (int i = 0; i < p.actual.length; i++)
		{
			int j = i * 2;
			if (pred.origIndex[j] + 1 != pred.origIndex[j + 1])
				throw new IllegalStateException();
			if (pred.actual[j] != pred.actual[j + 1])
				throw new IllegalStateException();
			if (pred.fold[j] != pred.fold[j + 1])
				throw new IllegalStateException();
			p.actual[i] = pred.actual[j];
			p.predicted[i] = (pred.predicted[j] + pred.predicted[j + 1]) / 2.0;
			p.confidence[i] = (pred.confidence[j] + pred.confidence[j + 1]) / 2.0;
			p.fold[i] = pred.fold[j];
			p.origIndex[i] = pred.origIndex[j] / 2;
		}

		return p;
	}

	public static double[] AUCPerFold(Predictions preds)
	{
		List<Double> pPerFold = new ArrayList<>();

		int minMax[] = ArrayUtil.getMinMax(preds.fold);
		for (int i = minMax[0]; i <= minMax[1]; i++)
		{
			List<Boolean> a = new ArrayList<>();
			List<Boolean> p = new ArrayList<>();
			List<Double> c = new ArrayList<>();
			for (int j = 0; j < preds.actual.length; j++)
			{
				if (preds.fold[j] == i)
				{
					a.add(preds.actual[j] == 1.0);
					p.add(preds.predicted[j] == 1.0);
					c.add(preds.confidence[j]);
				}
			}
			pPerFold.add(AUCComputer.compute(ArrayUtil.toPrimitiveBooleanArray(a),
					ArrayUtil.toPrimitiveBooleanArray(p), ArrayUtil.toPrimitiveDoubleArray(c)));
		}

		return ArrayUtil.toPrimitiveDoubleArray(pPerFold);
	}

	public static void printPredictionsWithWEKAProbability(Predictions preds,
			double positiveClassValue)
	{
		thresholdCurveInstances(preds, positiveClassValue, true);
	}

	protected static Instances thresholdCurveInstances(Predictions preds,
			final double positiveClassValue)
	{
		return thresholdCurveInstances(preds, positiveClassValue, false);
	}

	protected static Instances thresholdCurveInstances(Predictions preds,
			final double positiveClassValue, boolean print)
	{
		double probs[] = getProbabilitiesToBePositive(preds, positiveClassValue);

		ArrayList<Prediction> l = new ArrayList<>();
		for (int i = 0; i < preds.actual.length; i++)
		{
			double dist[];
			if (positiveClassValue == 0.0)
				dist = new double[] { probs[i], 1 - probs[i] };
			else if (positiveClassValue == 1.0)
				dist = new double[] { 1 - probs[i], probs[i] };
			else
				throw new IllegalStateException();
			l.add(new NominalPrediction(preds.actual[i], dist));
		}

		if (print)
		{
			ArrayList<Prediction> sort = new ArrayList<Prediction>(l);
			Collections.sort(sort, new Comparator<Prediction>()
			{
				@Override
				public int compare(Prediction o1, Prediction o2)
				{
					NominalPrediction n1 = (NominalPrediction) o1;
					NominalPrediction n2 = (NominalPrediction) o2;
					return Double.valueOf(n2.distribution()[(int) positiveClassValue])
							.compareTo(n1.distribution()[(int) positiveClassValue]);
				}
			});

			StringBuffer idx = new StringBuffer();
			StringBuffer conf = new StringBuffer();
			StringBuffer act = new StringBuffer();
			StringBuffer pred = new StringBuffer();
			int i = 0;
			double oldConf = -1;
			for (Prediction prediction : sort)
			{
				NominalPrediction n = (NominalPrediction) prediction;
				idx.append(String.format("%-5d", i));
				double newConf = n.distribution()[(int) positiveClassValue];
				conf.append(newConf != oldConf ? String.format("%-5.2f", newConf) : "     ");
				oldConf = newConf;
				act.append(String.format("%-5s", (n.actual() == positiveClassValue ? "x" : "-")));
				pred.append(
						String.format("%-5s", (n.predicted() == positiveClassValue ? "x" : "-")));
				i++;
			}
			System.out.println();
			System.out.println("idx:      " + idx);
			System.out.println("prob:     " + conf);
			System.out.println("act-pos:  " + act);
			System.out.println("pred-pos: " + pred);
			System.out.println();
		}

		ThresholdCurve tc = new ThresholdCurve();
		Instances inst = tc.getCurve(l, (int) positiveClassValue);
		//		System.out.println(inst);
		return inst;
	}

	public static double AUC(Predictions preds)
	{
		return ThresholdCurve.getROCArea(thresholdCurveInstances(preds, 0.0));
	}

	public static double AUPRC(Predictions preds, double postiveClassValue)
	{
		return ThresholdCurve.getPRCArea(thresholdCurveInstances(preds, postiveClassValue));
	}

	public static double accuracy(Predictions p)
	{
		int correct = 0;
		for (int i = 0; i < p.predicted.length; i++)
			if (p.predicted[i] == p.actual[i])
				correct++;
		return correct / (double) p.predicted.length;
	}

	public static double recall(Predictions p, double positiveClassValue)
	{
		return sensitivity(p, positiveClassValue);
	}

	public static double truePositiveRate(Predictions p, double positiveClassValue)
	{
		return sensitivity(p, positiveClassValue);
	}

	public static double sensitivity(Predictions p, double positiveClassValue)
	{
		double correct = 0, total = 0;
		for (int j = 0; j < p.actual.length; j++)
		{
			if (p.actual[j] == positiveClassValue)
			{
				if (p.predicted[j] == p.actual[j])
					correct++;
				total++;
			}
		}
		if (total == 0)
			return 0;
		return correct / total;
	}

	public static double trueNegativeRate(Predictions p, double positiveClassValue)
	{
		return specificity(p, positiveClassValue);
	}

	public static double specificity(Predictions p, double positiveClassValue)
	{
		double correct = 0, total = 0;
		for (int j = 0; j < p.actual.length; j++)
		{
			if (p.actual[j] != positiveClassValue)
			{
				if (p.predicted[j] == p.actual[j])
					correct++;
				total++;
			}
		}
		if (total == 0)
			return 0;
		return correct / total;
	}

	public static double enrichmentFactor(Predictions p, double percent, double positiveClassValue)
	{
		int allTotal = 0;
		int allClass = 0;
		for (int i = 0; i < p.actual.length; i++)
		{
			if (p.actual[i] == positiveClassValue)
				allClass++;
			allTotal++;
		}

		int topTotal = 0;
		int topClass = 0;
		Predictions top = topConfPositive(p, percent, positiveClassValue);
		//		System.err.println("X based on " + p.actual.length + " " + percent + " "
		//				+ (p.actual.length * percent));
		//		System.err.println("X based on " + top.actual.length);
		for (int i = 0; i < top.actual.length; i++)
		{
			if (top.actual[i] == positiveClassValue)
				topClass++;
			topTotal++;
		}

		double allRatio = allClass / (double) allTotal;
		double erRatio = topClass / (double) topTotal;
		return erRatio / allRatio;
	}

	public enum ClassificationMeasure
	{
		Accuracy, AUC, AUPRC, Sensitivity, Specificity, EnrichmentFactor5, EnrichmentFactor20,
		BEDROC20, BEDROC100;

		public static ClassificationMeasure[] SELECTION = { Accuracy, AUC, AUPRC, Sensitivity,
				Specificity };

		public String shortName()
		{
			switch (this)
			{
				case AUC:
					return "AUC";
				case AUPRC:
					return "AUPRC";
				case Accuracy:
					return "Accur";
				case Sensitivity:
					return "Sensi";
				case Specificity:
					return "Speci";
				case BEDROC20:
					return "BEDROC20";
				case BEDROC100:
					return "BEDROC100";
				case EnrichmentFactor5:
					return "EF5";
				case EnrichmentFactor20:
					return "EF20";
				default:
					throw new IllegalArgumentException();
			}
		}

	}

	public static double getClassificationMeasure(Predictions p, ClassificationMeasure m,
			double positiveClassValue)
	{
		switch (m)
		{
			case AUC:
				return AUC(p);
			case AUPRC:
				return AUPRC(p, positiveClassValue);
			case Accuracy:
				return accuracy(p);
			case Sensitivity:
				return sensitivity(p, positiveClassValue);
			case Specificity:
				return specificity(p, positiveClassValue);
			case EnrichmentFactor5:
				return enrichmentFactor(p, 0.05, positiveClassValue);
			case EnrichmentFactor20:
				return enrichmentFactor(p, 0.05, positiveClassValue);
			case BEDROC20:
				return toEnrichmentAssessment(p, positiveClassValue).bedroc(0.2, true);
			case BEDROC100:
				return toEnrichmentAssessment(p, positiveClassValue).bedroc(1, true);
			default:
				throw new IllegalArgumentException();
		}
	}

	public static double getClassificationMeasureInWeka(Evaluation eval, ClassificationMeasure m,
			double positiveClassValue)
	{
		switch (m)
		{
			case Accuracy:
				return eval.pctCorrect() / 100.0;
			case AUC:
				return eval.areaUnderROC((int) positiveClassValue);
			case AUPRC:
				return eval.areaUnderPRC((int) positiveClassValue);
			case Sensitivity:
				return eval.truePositiveRate((int) positiveClassValue);
			case Specificity:
				return eval.trueNegativeRate((int) positiveClassValue);
			default:
				throw new IllegalArgumentException(m + " does not exist in weka");
		}
	}

	public static double[] getProbabilitiesToBePositive(Predictions p, double positiveClassValue)
	{
		double probs[] = new double[p.actual.length];
		for (int i = 0; i < p.confidence.length; i++)
		{
			double prob = 0.5 + 0.5 * p.confidence[i];
			if (p.predicted[i] == positiveClassValue)
				probs[i] = prob;
			else
				probs[i] = 1 - prob;
		}
		return probs;
	}

	public static Predictions topConfPositive(Predictions p, double d, double positiveClassValue)
	{
		double probs[] = getProbabilitiesToBePositive(p, positiveClassValue);
		int scoreOrder[] = ArrayUtil.getOrdering(probs, false);
		return topFromOrdering(p, d, scoreOrder);
	}

	public static Predictions topConfAllClasses(Predictions pred, double d)
	{
		int confOrder[] = ArrayUtil.getOrdering(pred.confidence, false);
		return topFromOrdering(pred, d, confOrder);
	}

	private static Predictions topFromOrdering(Predictions pred, double d, int ordering[])
	{
		int size = (int) (pred.predicted.length * d);
		Predictions p = new Predictions();
		p.actual = new double[size];
		p.predicted = new double[size];
		p.confidence = new double[size];
		p.fold = new int[size];
		p.origIndex = new int[size];
		for (int i = 0; i < size; i++)
		{
			int j = ordering[i];
			p.actual[i] = pred.actual[j];
			p.predicted[i] = pred.predicted[j];
			p.confidence[i] = pred.confidence[j];
			p.fold[i] = pred.fold[j];
			p.origIndex[i] = pred.origIndex[j];
		}
		//		System.out.println("conf before:");
		//		System.out.println(DoubleArraySummary.create(pred.confidence).toStringSummary());
		//		System.out.println("conf after:");
		//		System.out.println(DoubleArraySummary.create(p.confidence).toStringSummary());
		return p;
	}

	public static Predictions stripActualNaN(Predictions p)
	{
		Predictions p2 = new Predictions();
		for (int i = 0; i < p.actual.length; i++)
			if (!Double.isNaN(p.actual[i]))
				add(p2, p, i);
		return p2;
	}

	/**
	 * @param s e.g. "1101101000100000100", 1=postive instance, 0 negative instance, sorted by prob
	 * @return
	 */
	public static Predictions fromBitString(String s)
	{
		String clazzes[] = ArrayUtil.toStringArray(ArrayUtils.toObject(s.toCharArray()));
		Predictions p = new Predictions();
		p.confidence = new double[clazzes.length];
		p.actual = new double[clazzes.length];
		p.predicted = new double[clazzes.length];
		p.fold = new int[clazzes.length];
		p.origIndex = new int[clazzes.length];
		double step = 1 / (double) (clazzes.length - 1);
		for (int i = 0; i < clazzes.length; i++)
		{
			p.confidence[i] = 1 - (i * step);
			p.actual[i] = clazzes[i].equals("1") ? 1.0 : 0.0;
			// predict all as active
			// otherwise the confidence has to computed more complicated:
			// it would have to go up when "moving to the right" of the bit-string
			p.predicted[i] = 1.0;
			p.fold[i] = -1;
			p.origIndex[i] = -1;
		}
		return p;
	}

	public static EnrichmentAssessment toEnrichmentAssessment(Predictions p,
			double positiveClassValue)
	{
		double probs[] = getProbabilitiesToBePositive(p, positiveClassValue);
		boolean labels[] = new boolean[p.actual.length];
		for (int i = 0; i < labels.length; i++)
			labels[i] = p.actual[i] == positiveClassValue;

		int ordering[] = ArrayUtil.getOrdering(probs, false);
		probs = ArrayUtil.sortAccordingToOrdering(ordering, probs);
		labels = ArrayUtil.sortAccordingToOrdering(ordering, labels);

		//		System.err.print("x <- c(");
		//		int i = 0;
		//		for (double prob : probs)
		//		{
		//			if (i > 0)
		//				System.err.print(",");
		//			if (i > 0 && i % 20 == 0)
		//				System.err.println();
		//			System.err.print(StringUtil.formatDouble(prob));
		//			i++;
		//		}
		//		System.err.println(")");
		//		System.err.print("y <- c(");
		//		i = 0;
		//		for (boolean b : labels)
		//		{
		//			if (i > 0)
		//				System.err.print(",");
		//			if (i > 0 && i % 20 == 0)
		//				System.err.println();
		//			System.err.print(b ? "1" : "0");
		//			i++;
		//		}
		//		System.err.println(")");
		//		System.exit(1);

		return new EnrichmentAssessment(probs, labels);
	}

	public static String summaryClassification(Predictions p)
	{
		StringBuffer bf = new StringBuffer();
		bf.append(p.actual.length + " predictions\n");
		bf.append("fold: " + CountedSet.create(ArrayUtil.toIntegerArray(p.fold)) + "\n");
		bf.append("actual: " + CountedSet.create(ArrayUtil.toDoubleArray(p.actual)) + "\n");
		bf.append("predicted: " + CountedSet.create(ArrayUtil.toDoubleArray(p.predicted)) + "\n");
		bf.append("confidence: " + DoubleArraySummary.create(p.confidence) + "\n");
		bf.append("accuracy: " + accuracy(p) + "\n");
		bf.append("auc: " + AUC(p) + "\n");
		return bf.toString();
	}

}
