package org.mg.wekalib.eval2;

import java.io.FileReader;
import java.io.Serializable;
import java.lang.reflect.Array;

import org.mg.javalib.util.ArrayUtil;
import org.mg.wekalib.eval2.data.DataSet;
import org.mg.wekalib.eval2.data.WekaInstancesDataSet;
import org.mg.wekalib.eval2.job.DataSetJobOwner;
import org.mg.wekalib.eval2.job.DefaultJobOwner;
import org.mg.wekalib.eval2.job.JobOwner;
import org.mg.wekalib.eval2.job.Printer;
import org.mg.wekalib.eval2.model.RandomForestModel;
import org.mg.wekalib.evaluation.PredictionUtil;
import org.mg.wekautil.Predictions;

import weka.core.Instances;

public class MultiDatasetRunner<R extends Serializable> extends DefaultJobOwner<R[]>
{
	DataSet dataSets[];
	DataSetJobOwner<R> job;

	public void setJob(DataSetJobOwner<R> job)
	{
		this.job = job;
	}

	public void setDataSets(DataSet... dataSets)
	{
		this.dataSets = dataSets;
	}

	@Override
	public JobOwner<R[]> cloneJob()
	{
		MultiDatasetRunner<R> r = new MultiDatasetRunner<>();
		r.setJob((DataSetJobOwner<R>) job.cloneJob());
		r.setDataSets(dataSets);
		return r;
	}

	@Override
	public String getKey()
	{
		return getKey(job, dataSets);
	}

	@Override
	public Runnable nextJob() throws Exception
	{
		boolean allDone = true;
		for (DataSet d : dataSets)
		{
			DataSetJobOwner<R> job = (DataSetJobOwner<R>) this.job.cloneJob();
			job.setDataSet(d);
			if (!job.isDone())
			{
				allDone = false;
				Runnable r = job.nextJob();
				if (r != null)
				{
					return Printer.wrapRunnable("MultiDataset: run " + (ArrayUtil.indexOf(dataSets, d) + 1) + "/"
							+ dataSets.length + " with dataset " + d.getName(), r);
				}
			}
		}

		if (allDone)
		{
			return blockedJob("MultiDataset: storing results", new Runnable()
			{
				@Override
				public void run()
				{
					store();
				}
			});
		}
		else
			return null;
	}

	@SuppressWarnings("unchecked")
	private void store()
	{
		R resultArray[] = null;
		int idx = 0;
		for (DataSet d : dataSets)
		{
			DataSetJobOwner<R> job = (DataSetJobOwner<R>) this.job.cloneJob();
			job.setDataSet(d);
			R result = job.getResult();
			if (resultArray == null)
				resultArray = (R[]) Array.newInstance(result.getClass(), dataSets.length);
			resultArray[idx++] = result;
		}
		setResult(resultArray);
	}

	public static void main(String[] args) throws Exception
	{
		Instances inst = new Instances(new FileReader("/home/martin/data/weka/nominal/breast-w.arff"));
		Instances inst2 = new Instances(new FileReader("/home/martin/data/weka/nominal/credit-a.arff"));
		inst.setClassIndex(inst.numAttributes() - 1);
		inst2.setClassIndex(inst2.numAttributes() - 1);
		MultiDatasetRunner<Predictions> run = new MultiDatasetRunner<>();
		run.setDataSets(new WekaInstancesDataSet(inst), new WekaInstancesDataSet(inst2));
		CV cv = new CV();
		cv.setModel(new RandomForestModel());
		cv.setNumFolds(5);
		run.setJob(cv);
		run.runSequentially();
		for (Predictions p : run.getResult())
			System.out.println(PredictionUtil.summaryClassification(p));
	}

}
