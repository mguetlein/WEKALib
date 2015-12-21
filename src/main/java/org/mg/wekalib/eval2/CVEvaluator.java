package org.mg.wekalib.eval2;

import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;

import org.mg.javalib.datamining.ResultSet;
import org.mg.wekalib.eval2.data.DataSet;
import org.mg.wekalib.eval2.data.WekaInstancesDataSet;
import org.mg.wekalib.eval2.job.DataSetJobOwner;
import org.mg.wekalib.eval2.job.DefaultJobOwner;
import org.mg.wekalib.eval2.job.Printer;
import org.mg.wekalib.eval2.model.Model;
import org.mg.wekalib.eval2.model.NaiveBayesModel;
import org.mg.wekalib.eval2.model.RandomForestModel;
import org.mg.wekalib.evaluation.PredictionUtil;
import org.mg.wekautil.Predictions;

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

	public CVEvaluator cloneJob()
	{
		CVEvaluator cv = new CVEvaluator();
		cv.setModels(models);
		cv.setDataSet(dataSet);
		cv.setNumFolds(numFolds);
		cv.setRepetitions(repetitions);
		return cv;
	}

	@Override
	public String getKey()
	{
		return getKey(numFolds, repetitions, dataSet, models);
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
					return Printer.wrapRunnable("CVEval: cv " + ((getCVs().indexOf(cv)) + 1) + "/" + getCVs().size(),
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
					rs.setResultValue(idx, "ModelKey", cv.getModel().getKey());
					rs.setResultValue(idx, "ModelName", cv.getModel().getName());
					rs.setResultValue(idx, "CVSeed", cv.getRandomSeed());
					rs.setResultValue(idx, "CVFold", p.fold[0]);
					rs.setResultValue(idx, "AUC", PredictionUtil.AUC(p));
				}
			}
		}

		rs = rs.join("ModelKey");
		String maxAucK = null;
		double maxAucV = 0.0;
		for (int i = 0; i < rs.getNumResults(); i++)
		{
			double auc = (Double) rs.getResultValue(i, "AUC");
			if (auc > maxAucV)
			{
				maxAucV = auc;
				maxAucK = rs.getResultValue(i, "ModelKey").toString();
			}
		}
		//System.err.println("results:\n" + rs.toNiceString());
		setResult(maxAucK);
		Model m = getBestModel();
		Printer.println("best model:\n" + m.getName());
	}

	public Model getBestModel()
	{
		String result = getResult();
		Model m = null;
		for (CV cv : getCVs())
		{
			if (cv.getModel().getKey().equals(result))
			{
				m = cv.getModel();
				break;
			}
		}
		return m;
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

	public static void main(String[] args) throws Exception
	{
		final CVEvaluator cv = new CVEvaluator();
		Instances inst = new Instances(new FileReader("/home/martin/data/weka/nominal/breast-w.arff"));
		inst.setClassIndex(inst.numAttributes() - 1);
		cv.setDataSet(new WekaInstancesDataSet(inst));
		cv.setModels(new RandomForestModel(), new NaiveBayesModel());
		cv.setNumFolds(10);
		cv.setRepetitions(1);
		cv.runSequentially();
		System.out.println("done: " + cv.getBestModel());
	}

}
