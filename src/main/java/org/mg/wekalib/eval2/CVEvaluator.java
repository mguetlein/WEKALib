package org.mg.wekalib.eval2;

import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;

import org.mg.javalib.datamining.ResultSet;
import org.mg.javalib.util.ArrayUtil;
import org.mg.javalib.util.HashUtil;
import org.mg.wekalib.evaluation.PredictionUtil;
import org.mg.wekautil.Predictions;

import weka.core.Instances;

public class CVEvaluator extends DefaultJobOwner<Integer>
{
	DataSet dataSet;
	Model models[];
	//	FeatureProvider featureProviders[];
	int numFolds = 10;
	int repetitions = 3;

	public CVEvaluator cloneCVEvaluator()
	{
		CVEvaluator cv = new CVEvaluator();
		cv.setModels(models);
		cv.setDataSet(dataSet);
		cv.setNumFolds(numFolds);
		cv.setRepetitions(repetitions);
		return cv;
	}

	@Override
	public int hashCode()
	{
		Object[] os = new Object[] { numFolds, repetitions, dataSet };
		os = ArrayUtil.concat(Object.class, os, models);
		//		if (featureProviders != null)
		//			os = ArrayUtil.concat(Object.class, os, featureProviders);
		return HashUtil.hashCode(os);
	}

	private List<CV> cvs;

	private List<CV> getCVs()
	{
		if (cvs == null)
		{
			cvs = new ArrayList<>();
			for (int r = 0; r < repetitions; r++)
			{
				for (Model mod : models)
				{
					//					for (FeatureProvider feat : (featureProviders != null ? featureProviders
					//							: new FeatureProvider[] { null }))
					//					{
					CV cv = new CV();
					cv.setModel(mod.cloneModel());
					//						if (feat != null)
					//							cv.setFeatureProvider(feat.cloneFeatureProvider());
					cv.setDataSet(dataSet);
					cv.setRandomSeed((long) r);
					cv.setNumFolds(numFolds);
					cvs.add(cv);
					//					}
				}
			}
		}
		return cvs;
	}

	@Override
	public Runnable nextJob() throws Exception
	{
		boolean allDone = true;
		for (final CV cv : getCVs())
		{
			if (!cv.isDone())
			{
				allDone = false;
				final Runnable run = cv.nextJob();
				if (run != null)
					return run;
			}
		}
		if (allDone)
		{
			if (!Blocker.block(hashCode()))
				return null;
			return new Runnable()
			{
				@Override
				public void run()
				{
					System.out.println(CVEvaluator.this.hashCode() + " storing cv evaluation");
					store();
					Blocker.unblock(CVEvaluator.this.hashCode());
				}
			};
		}
		return null;
	}

	protected void store()
	{
		ResultSet rs = new ResultSet();

		for (CV cv : getCVs())
		{
			if (!cv.isDone())
				throw new IllegalArgumentException("cv no yet done!");
			else
			{
				for (Predictions p : PredictionUtil.perFold(cv.getResult()))
				{
					int idx = rs.addResult();
					rs.setResultValue(idx, "ModelKey", cv.getModel().hashCode());
					rs.setResultValue(idx, "ModelName", cv.getModel().getName());
					//					if (cv.getFeatureProvider() != null)
					//					{
					//						rs.setResultValue(idx, "FeatureKey", cv.getFeatureProvider().hashCode());
					//						rs.setResultValue(idx, "FeatureName", cv.getFeatureProvider().getName());
					//					}
					rs.setResultValue(idx, "CVSeed", cv.getRandomSeed());
					rs.setResultValue(idx, "CVFold", p.fold[0]);
					rs.setResultValue(idx, "AUC", PredictionUtil.AUC(p));
				}
			}
		}

		rs = rs.join("ModelKey");
		int maxAucK = -1;
		double maxAucV = 0.0;
		for (int i = 0; i < rs.getNumResults(); i++)
		{
			double auc = (Double) rs.getResultValue(i, "AUC");
			if (auc > maxAucV)
			{
				maxAucV = auc;
				maxAucK = (Integer) rs.getResultValue(i, "ModelKey");
			}
		}
		System.err.println("results:\n" + rs.toNiceString());
		setResult(maxAucK);
		getBestModel();
	}

	public Model getBestModel()
	{
		Integer result = getResult();
		Model m = null;
		for (CV cv : getCVs())
		{
			if (cv.getModel().hashCode() == result)
			{
				m = cv.getModel();
				break;
			}
		}
		System.err.println("max model:\n" + m.getName());
		return m;
	}

	public void setDataSet(DataSet dataSet)
	{
		this.dataSet = dataSet;
	}

	public void setModels(Model... models)
	{
		this.models = models;
	}

	//	public void setFeatureProviders(FeatureProvider... feats)
	//	{
	//		this.featureProviders = feats;
	//	}

	public void setNumFolds(int numFolds)
	{
		this.numFolds = numFolds;
	}

	public void setRepetitions(int repetitions)
	{
		this.repetitions = repetitions;
	}

	public static void main(String[] args) throws Exception
	{
		final CVEvaluator cv = new CVEvaluator();
		Instances inst = new Instances(new FileReader("/home/martin/data/weka/nominal/breast-w.arff"));
		inst.setClassIndex(inst.numAttributes() - 1);
		cv.setDataSet(new WekaInstancesDataSet(inst));
		cv.setModels(new RandomForestModel(), new NaiveBayesModel());
		cv.setNumFolds(10);
		cv.setRepetitions(1);
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
		//		if (cv.isDone())
		//			System.out.println(PredictionUtil.summaryClassification(cv.getResult()));
	}

}
