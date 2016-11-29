package org.mg.wekalib.eval2.data;

import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVPrinter;
import org.apache.commons.csv.CSVRecord;
import org.apache.commons.io.IOUtils;
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.mg.javalib.util.ArrayUtil;
import org.mg.javalib.util.ListUtil;
import org.mg.wekalib.distance.Distance;
import org.mg.wekalib.distance.TanimotoDistance;
import org.mg.wekalib.eval2.job.DefaultComposedKeyProvider;

import weka.core.Instances;
import weka.core.converters.CSVSaver;

public abstract class WekaAntiStratifiedSplitter extends DefaultComposedKeyProvider
		implements AntiStratifiedSplitter
{
	public static Instances[] split(Instances data, double ratio)
	{
		Instances inst = new Instances(data);
		inst.randomize(new Random());
		int trainSize = (int) Math.round(inst.numInstances() * ratio);
		int testSize = inst.numInstances() - trainSize;
		Instances train = new Instances(inst, 0, trainSize);
		Instances test = new Instances(inst, trainSize, testSize);
		return new Instances[] { train, test };
	}

	@Override
	public String getKeyContent()
	{
		return getKeyContent("");
	}

	@Override
	public String getKeyPrefix()
	{
		return getKeyContent();
	}

	public static List<Integer> antiStratifiedSplitIndices(Instances data, double ratio,
			final Distance dist, long seed, boolean train)
	{
		WekaAntiStratifiedSplitter split = new WekaAntiStratifiedSplitter()
		{
			@Override
			public Distance getDistance()
			{
				return dist;
			}
		};
		return split.antiStratifiedSplitIndices(new WekaInstancesDataSet(data, -1), ratio, seed,
				train);
	}

	public abstract Distance getDistance();

	public List<Integer> antiStratifiedSplitIndices(DataSet data, double ratio, long seed,
			boolean train)
	{
		Random r = new Random(seed);
		Instances inst = data.getWekaInstances();
		//inst.randomize(r);
		Distance dist = getDistance();
		dist.build(inst);

		// step 1: determine centroid
		System.err.println("determine centroid");
		int indices[] = ArrayUtil.indexArray(inst.size());
		ArrayUtil.scramble(indices, r);
		int numCentroidCandidates = (int) Math.min(inst.numInstances(),
				Math.max(10, inst.numInstances() * 0.005));
		double maxDist = 0;
		int centroidIdx = -1;
		for (int j = 0; j < numCentroidCandidates; j++)
		{
			int i = indices[j];
			if (dist instanceof TanimotoDistance && dist.distance(i, i) == 1.0)
			{
				// if a fragment has not active bits, it has distance 1 to itself
				// do not use as centroid, no information here
				continue;
			}
			DescriptiveStatistics stats = new DescriptiveStatistics();
			for (int k = 0; k < inst.size(); k++)
			{
				if (i == k || r.nextDouble() < 0.66)
					continue;
				stats.addValue(dist.distance(i, k));
			}
			double d = stats.getMean();
			//			System.out.println(i + " " + d);
			if (d > maxDist)
			{
				maxDist = d;
				centroidIdx = i;
			}
			//			else if (d == maxDist)
			//				System.out.println("equal");
		}

		//		System.out.println("centroid " + centroidIdx + " " + maxDist);

		// step 2: compute similarity to centroid
		System.err.println("compute similarity to centroid");
		double distToCentroid[] = new double[inst.size()];
		double maxDistToCentroid = 0;
		for (int i = 0; i < distToCentroid.length; i++)
		{
			if (centroidIdx == i)
			{
				if (distToCentroid[i] != 0)
					throw new IllegalStateException(dist.distance(centroidIdx, i) + "");
			}
			else
			{
				distToCentroid[i] = dist.distance(centroidIdx, i);
				maxDistToCentroid = Math.max(distToCentroid[i], maxDistToCentroid);
			}
		}
		double simToCentroid[] = new double[inst.size()];
		double simSum = 0;
		for (int i = 0; i < simToCentroid.length; i++)
		{
			if (centroidIdx == i)
				continue;
			simToCentroid[i] = Math.pow(maxDistToCentroid - distToCentroid[i], 10);
			simSum += simToCentroid[i];
		}

		// step3: sample first set by probability
		System.err.println("sample with probability");
		int sampleSize = (int) (inst.numInstances() * ratio);
		List<Integer> selected = new ArrayList<>();
		Set<Integer> available = new HashSet<>(
				ArrayUtil.toList(ArrayUtil.indexArray(inst.numInstances())));
		selected.add(centroidIdx);
		available.remove(centroidIdx);

		//		{
		//			System.out.println(simSum + " initial sum");
		//			double distSum2 = 0;
		//			for (Integer idx : available)
		//				distSum2 += simToCentroid[idx];
		//			distSum2 += 0;
		//			System.out.println(distSum2 + " check sum");
		//			if (distSum2 != simSum)
		//				throw new IllegalStateException();
		//
		//			System.out.println("centroid " + centroidIdx);
		//			System.out.println("centroid " + simToCentroid[centroidIdx]);
		//		}

		while (selected.size() < sampleSize)
		{
			double p = r.nextDouble() * simSum;
			double cumulativeProbability = 0.0;
			int selectedIdx = -1;
			for (Integer idx : available)
			{
				cumulativeProbability += simToCentroid[idx];
				if (p <= cumulativeProbability)
				{
					//					System.out.println("select " + idx);
					//					System.out.println("select " + simToCentroid[idx]);
					selectedIdx = idx;
					// do re-calc sum with addition instead of substraction
					// otherwise floating point in-accuracies lead to problems
					//simSum -= simToCentroid[idx];
					break;
				}
			}
			//			System.out.println(simSum + " sum after select");

			if (selectedIdx == -1)
				throw new IllegalStateException();
			selected.add(selectedIdx);
			available.remove(selectedIdx);

			simSum = 0;
			for (Integer idx : available)
				simSum += simToCentroid[idx];
		}

		if (train)
		{
			ListUtil.scramble(selected, r);
			return selected;
		}
		else
		{
			List<Integer> unselected = new ArrayList<>(available);
			ListUtil.scramble(unselected, r);
			return unselected;
		}
	}

	public static Instances[] antiStratifiedSplit(Instances data, double ratio, Distance dist,
			long seed) throws Exception
	{
		List<Integer> selected = antiStratifiedSplitIndices(data, ratio, dist, seed, true);

		// create training and test set
		Instances train = new Instances(data, 0);
		Instances test = new Instances(data, 0);
		for (int i = 0; i < data.numInstances(); i++)
		{
			if (selected.contains(i))
				train.add(data.get(i));
			else
				test.add(data.get(i));
		}

		return new Instances[] { train, test };
	}

	public static void main(String[] args) throws Exception
	{
		CSVParser p1 = null;
		CSVParser p2 = null;
		CSVPrinter print = null;
		try
		{
			String data = "CPDBAS_Mouse";
			Instances inst = new Instances(new FileReader(
					System.getProperty("user.home") + "/data/weka/nominal/" + data + ".arff"));
			inst.setClassIndex(inst.numAttributes() - 1);
			inst.randomize(new Random(123));

			Instances split[] = antiStratifiedSplit(inst, 0.5, new TanimotoDistance(), 123);

			System.out.println(split[0].numInstances());

			CSVSaver s = new CSVSaver();
			s.setDestination(new FileOutputStream("/tmp/train.csv"));
			s.setInstances(split[0]);
			s.writeBatch();

			s = new CSVSaver();
			s.setDestination(new FileOutputStream("/tmp/test.csv"));
			s.setInstances(split[1]);
			s.writeBatch();

			p1 = new CSVParser(new FileReader("/tmp/train.csv"), CSVFormat.RFC4180.withHeader());
			p2 = new CSVParser(new FileReader("/tmp/test.csv"), CSVFormat.RFC4180.withHeader());
			if (!p1.getHeaderMap().equals(p2.getHeaderMap()))
				throw new IllegalStateException();

			print = new CSVPrinter(new FileWriter("/tmp/merged.csv"), CSVFormat.RFC4180);

			List<String> header = new ArrayList<>();
			header.add("SMILES");
			header.addAll(p1.getHeaderMap().keySet());
			header.add("Dataset");
			print.printRecord(header);

			for (CSVRecord r : p1.getRecords())
			{
				List<String> values = new ArrayList<>();
				values.add("C");
				for (String k : p1.getHeaderMap().keySet())
					values.add(r.get(k));
				values.add("train");
				print.printRecord(values);
			}
			for (CSVRecord r : p2.getRecords())
			{
				List<String> values = new ArrayList<>();
				values.add("C");
				for (String k : p1.getHeaderMap().keySet())
					values.add(r.get(k));
				values.add("test");
				print.printRecord(values);
			}
		}
		finally
		{
			IOUtils.closeQuietly(print);
			IOUtils.closeQuietly(p1);
			IOUtils.closeQuietly(p2);
		}

		//		for (CSVRecord record : p.getRecords())
		//		{
		//			String id = record.get("ID");
		//			String customerNo = record.get("CustomerNo");
		//			String name = record.get("Name");
		//		}
	}

}
