package org.mg.wekalib.eval2;

import java.io.File;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;

import org.mg.javalib.datamining.ResultSet;
import org.mg.wekalib.eval2.data.DataSet;
import org.mg.wekalib.eval2.data.WekaInstancesDataSet;
import org.mg.wekalib.eval2.job.ComposedKeyProvider;
import org.mg.wekalib.eval2.job.DataSetJobOwner;
import org.mg.wekalib.eval2.job.DefaultJobOwner;
import org.mg.wekalib.eval2.job.Printer;
import org.mg.wekalib.eval2.model.Model;
import org.mg.wekalib.eval2.model.NaiveBayesModel;
import org.mg.wekalib.eval2.model.RandomForestModel;
import org.mg.wekalib.evaluation.PredictionUtil;
import org.mg.wekalib.evaluation.Predictions;

import weka.core.Instances;

/**
 * repeats single cross-validations ({@link CV}) with a range of models and multiple repetitions
 * result: key of best model
 */
public class CVEvaluator extends DefaultJobOwner<String> implements DataSetJobOwner<String>
{
	DataSet dataSet;
	Model models[];
	int numFolds = 10;
	int repetitions = 3;
	EvalCriterion crit = new DefaultEvalCriterion(PredictionUtil.ClassificationMeasure.AUPRC);

	public CVEvaluator cloneJob()
	{
		CVEvaluator cv = new CVEvaluator();
		cv.setModels(models);
		cv.setDataSet(dataSet);
		cv.setNumFolds(numFolds);
		cv.setRepetitions(repetitions);
		cv.setEvalCriterion(crit);
		return cv;
	}

	@Override
	public String getName()
	{
		return "CVEvaluator";
	}

	public static interface EvalCriterion extends ComposedKeyProvider
	{
		public String selectBestModel(List<CV> cvs);
	}

	public static class DefaultEvalCriterion implements EvalCriterion
	{
		PredictionUtil.ClassificationMeasure measure;

		public DefaultEvalCriterion(PredictionUtil.ClassificationMeasure measure)
		{
			this.measure = measure;
		}

		@Override
		public String getKeyPrefix()
		{
			return measure.toString();
		}

		@Override
		public String getKeyContent()
		{
			return measure.toString();
		}

		@Override
		public String selectBestModel(List<CV> cvs)
		{
			ResultSet rs = new ResultSet();
			for (CV cv : cvs)
			{
				if (!cv.isDone())
					throw new IllegalArgumentException("cv no yet done!");
				else
				{
					for (Predictions p : PredictionUtil.perFold(cv.getResult()))
					{
						int idx = rs.addResult();
						rs.setResultValue(idx, "ModelKey", cv.getModel().getKey());
						rs.setResultValue(idx, "ModelName", cv.getModel().getName());
						rs.setResultValue(idx, "CVSeed", cv.getRandomSeed());
						rs.setResultValue(idx, "CVFold", p.fold[0]);
						rs.setResultValue(idx, measure.toString(),
								PredictionUtil.getClassificationMeasure(p, measure,
										cv.getDataset().getPositiveClass()));
					}
				}
			}
			rs = rs.join(new String[] { "ModelKey", "ModelName" }, null, null);
			String maxAucK = null;
			double maxAucV = 0.0;
			for (int i = 0; i < rs.getNumResults(); i++)
			{
				double auc = (Double) rs.getResultValue(i, measure.toString());
				if (auc > maxAucV)
				{
					maxAucV = auc;
					maxAucK = rs.getResultValue(i, "ModelKey").toString();
				}
			}
			return maxAucK;
		}
	}

	@Override
	public String getKeyPrefix()
	{
		return "CVEvaluator-numFolds" + numFolds + "-repetitions" + repetitions + "-measure"
				+ crit.getKeyPrefix()
				+ (dataSet != null ? (File.separator + dataSet.getKeyPrefix()) : "");
	}

	@Override
	public String getKeyContent()
	{
		return getKeyContent(numFolds, repetitions, crit.getKeyContent(), dataSet, models);
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
					CV cv = new CV();
					cv.setModel((Model) mod.cloneJob());
					cv.setDataSet(dataSet);
					cv.setRandomSeed((long) r);
					cv.setNumFolds(numFolds);
					cvs.add(cv);
				}
			}
		}
		return cvs;
	}

	@Override
	public Runnable nextJob() throws Exception
	{
		if (dataSet == null)
			throw new NullPointerException("set dataset first");

		boolean allDone = true;
		for (final CV cv : getCVs())
		{
			if (!cv.isDone())
			{
				allDone = false;
				final Runnable run = cv.nextJob();
				if (run != null)
					return Printer.wrapRunnable(
							"CVEval: cv " + ((getCVs().indexOf(cv)) + 1) + "/" + getCVs().size(),
							run);
			}
		}

		if (allDone)
			return blockedJob("CVEval: storing results", new Runnable()
			{
				@Override
				public void run()
				{
					store();
				}
			});
		else
			return null;
	}

	protected void store()
	{
		String bestModelKey = crit.selectBestModel(getCVs());
		setResult(bestModelKey);
		Model m = getBestModel();
		Printer.println("best model:\n" + m.getName());
	}

	public List<Predictions> getPredictions()
	{
		List<Predictions> l = new ArrayList<Predictions>();
		for (CV cv : getCVs())
		{
			//			System.out.println(cv.getName() + " " + cv.getModel().getName());
			if (!cv.isDone())
				throw new IllegalArgumentException("cv no yet done!");
			else
				l.add(cv.getResult());
		}
		if (l.size() != repetitions * models.length)
			throw new IllegalArgumentException();
		return l;
	}

	public Model getBestModel()
	{
		String result = getResult();
		for (CV cv : getCVs())
			if (cv.getModel().getKey().toString().equals(result))
				return cv.getModel();
		throw new IllegalStateException("key does not fit");
	}

	@Override
	public void setDataSet(DataSet dataSet)
	{
		this.dataSet = dataSet;
	}

	public void setModels(Model... models)
	{
		this.models = models;
	}

	public void setNumFolds(int numFolds)
	{
		this.numFolds = numFolds;
	}

	public void setRepetitions(int repetitions)
	{
		this.repetitions = repetitions;
	}

	public void setEvalCriterion(EvalCriterion crit)
	{
		this.crit = crit;
	}

	public static void main(String[] args) throws Exception
	{
		final CVEvaluator cv = new CVEvaluator();
		Instances inst = new Instances(
				new FileReader("/home/martin/data/weka/nominal/breast-w.arff"));
		inst.setClassIndex(inst.numAttributes() - 1);
		cv.setDataSet(new WekaInstancesDataSet(inst, 1));
		cv.setModels(new RandomForestModel(), new NaiveBayesModel());
		cv.setNumFolds(10);
		cv.setRepetitions(1);
		cv.runSequentially();
		System.out.println("done: " + cv.getBestModel());
	}

	public DataSet getDataset()
	{
		return dataSet;
	}

}
