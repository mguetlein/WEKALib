package org.mg.wekalib.eval2;

import java.io.FileReader;

import org.mg.javalib.util.HashUtil;
import org.mg.wekalib.evaluation.PredictionUtil;
import org.mg.wekautil.Predictions;

import weka.core.Instances;

public class CV extends DefaultJobOwner<Predictions>
{
	DataSet d;
	//	FeatureProvider featureProvider;
	Model m;
	int numFolds = 10;
	long randomSeed = 1;

	public CV cloneCV()
	{
		CV cv = new CV();
		cv.setDataSet(d);
		cv.setModel(m);
		//		cv.setFeatureProvider(featureProvider);
		cv.setNumFolds(numFolds);
		cv.setRandomSeed(randomSeed);
		return cv;
	}

	//	private FeatureProvider getFeatureProvider(int fold)
	//	{
	//		FeatureProvider feat = featureProvider.cloneFeatureProvider();
	//		feat.setTrainingDataset(d.getTrainFold(numFolds, randomSeed, fold));
	//		feat.setTestDataset(d.getTestFold(numFolds, randomSeed, fold));
	//		return feat;
	//	}

	private Model getModel(int fold)
	{
		Model mod = m.cloneModel();
		//		if (featureProvider != null)
		//		{
		//			DataSet res[] = getFeatureProvider(fold).getResult();
		//			mod.setTrainingDataset(res[0]);
		//			mod.setTestDataset(res[1]);
		//		}
		//		else
		//		{
		mod.setTrainingDataset(d.getTrainFold(numFolds, randomSeed, fold));
		mod.setTestDataset(d.getTestFold(numFolds, randomSeed, fold));
		//		}
		return mod;
	}

	@Override
	public int hashCode()
	{
		return HashUtil.hashCode(d, m, /*featureProvider,*/numFolds, randomSeed);
	}

	@Override
	public Runnable nextJob() throws Exception
	{
		// run cv
		boolean allDone = true;
		for (int f = 0; f < numFolds; f++)
		{
			//			if (featureProvider != null)
			//			{
			//				FeatureProvider feat = getFeatureProvider(f);
			//				if (!feat.isDone())
			//				{
			//					allDone = false;
			//					Runnable r = feat.nextJob();
			//					if (r != null)
			//						return r;
			//					else
			//						continue;
			//				}
			//			}

			Model mod = getModel(f);
			if (!mod.isDone())
			{
				//				System.out.println(model.getTrainingDataset());
				//				System.out.println(model.getFeatureProvider().getTrainingDataset());
				allDone = false;
				Runnable r = mod.nextJob();
				if (r != null)
					return r;
			}
		}

		if (allDone)
		{
			if (!Blocker.block(CV.this.hashCode()))
				return null;
			return new Runnable()
			{
				@Override
				public void run()
				{
					System.out.println(CV.this.hashCode() + " storing cv results");
					store();
					Blocker.unblock(CV.this.hashCode());
				}
			};
		}
		return null;
	}

	protected void store()
	{
		Predictions pred = new Predictions();
		for (int f = 0; f < numFolds; f++)
		{
			Predictions p = getModel(f).getResult();
			for (int i = 0; i < p.actual.length; i++)
				p.fold[i] = f;
			pred = PredictionUtil.concat(pred, p);
		}
		System.err.println(PredictionUtil.summaryClassification(pred));
		setResult(pred);
	}

	public Model getModel()
	{
		return m;
	}

	public void setModel(Model mod)
	{
		m = mod;
	}

	public void setDataSet(DataSet data)
	{
		d = data;
	}

	//	public void setFeatureProvider(FeatureProvider featureProvider)
	//	{
	//		this.featureProvider = featureProvider;
	//	}
	//
	//	public FeatureProvider getFeatureProvider()
	//	{
	//		return featureProvider;
	//	}

	public long getRandomSeed()
	{
		return randomSeed;
	}

	public void setRandomSeed(long r)
	{
		randomSeed = r;
	}

	public void setNumFolds(int numFolds)
	{
		this.numFolds = numFolds;
	}

	public static void main(String[] args) throws Exception
	{
		final CV cv = new CV();
		Instances inst = new Instances(new FileReader("/home/martin/data/weka/nominal/breast-w.arff"));
		inst.setClassIndex(inst.numAttributes() - 1);
		cv.setDataSet(new WekaInstancesDataSet(inst));
		cv.setModel(new RandomForestModel());
		if (!cv.isDone())
		{
			Runnable r = cv.nextJob();
			while (r != null)
			{
				r.run();
				if (!cv.isDone())
					r = cv.nextJob();
				else
					r = null;
			}
		}
		System.out.println(cv.hashCode());
		if (cv.isDone())
			System.out.println(PredictionUtil.summaryClassification(cv.getResult()));

		//		for (int i = 0; i < cv.numFolds; i++)
		//		{
		//			Thread th = new Thread(new Runnable()
		//			{
		//				@Override
		//				public void run()
		//				{
		//					try
		//					{
		//						CV c = cv.cloneCV();
		//						if (!c.isDone())
		//						{
		//							Runnable r = c.nextJob();
		//							if (r != null)
		//								r.run();
		//						}
		//					}
		//					catch (Exception e)
		//					{
		//						throw new RuntimeException(e);
		//					}
		//				}
		//			});
		//			th.start();
		//			//			ThreadUtil.sleep(100);
		//		}
		//		while (!cv.isDone())
		//		{
		//			ThreadUtil.sleep(100);
		//			System.out.println("not yet done");
		//		}
		//		System.out.println(PredictionUtil.summaryClassification(cv.getResult()));
	}
}
