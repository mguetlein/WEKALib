package org.mg.wekalib.appdomain;

import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;

import weka.core.EuclideanDistance;
import weka.core.Instance;
import weka.core.Instances;

public class AppDomainTest
{
	public static void main(String[] args) throws FileNotFoundException, IOException
	{
		String data = "vote";
		Instances inst = new Instances(new FileReader(
				System.getProperty("user.home") + "/data/weka/nominal/" + data + ".arff"));
		inst.setClassIndex(inst.numAttributes() - 1);
		//System.out.println(inst);

		NNDistanceBasedAppDomainModel ad = new NNDistanceBasedAppDomainModel()
		//DistanceDistributionBasedAppDomainModel ad = new DistanceDistributionBasedAppDomainModel()
		{
			EuclideanDistance d;

			@Override
			public double computeDistance(Instance i1, Instance i2)
			{
				return d.distance(i1, i2);
			}

			@Override
			public void buildInternal(Instances trainingData)
			{
				d = new EuclideanDistance(trainingData);
			}
		};
		ad = new TanimotoAppDomainModel();
		ad.build(inst);

		int inside = 0;
		//		List<Double> probInside = new ArrayList<>();
		for (int i = 0; i < inst.size(); i++)
		{
			if (ad.isInsideAppdomain(inst.get(i)))
				inside++;
			//			probInside.add(ad.insideProbability(inst.get(i)));
		}
		System.out.println(inside + "/" + inst.size() + " inside");
		//		System.out.println(DoubleArraySummary.create(probInside));
	}
}
