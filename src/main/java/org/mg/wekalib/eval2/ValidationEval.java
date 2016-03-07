package org.mg.wekalib.eval2;

import java.io.File;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;

import org.mg.javalib.datamining.ResultSet;
import org.mg.javalib.util.ListUtil;
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
 * repeats single cross-validations or hold-outs ({@link Validation}) with a range of models and multiple repetitions
 * result: key of best model
 */
public class ValidationEval extends DefaultJobOwner<String> implements DataSetJobOwner<String>
{
	DataSet dataSet;
	Model models[];
	int repetitions = 3;
	Validation validation;
	EvalCriterion crit = new DefaultEvalCriterion(PredictionUtil.ClassificationMeasure.AUPRC);

	public ValidationEval cloneJob()
	{
		ValidationEval v = new ValidationEval();
		v.setModels(models);
		v.setDataSet(dataSet);
		v.setRepetitions(repetitions);
		v.setEvalCriterion(crit);
		v.setValidation((Validation) validation.cloneJob());
		return v;
	}

	@Override
	public String getName()
	{
		return this.getClass().getSimpleName();
	}

	public static interface EvalCriterion extends ComposedKeyProvider
	{
		public String selectBestModel(List<Validation> validations);
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
		public String selectBestModel(List<Validation> validations)
		{
			ResultSet rs = new ResultSet();
			for (Validation v : validations)
			{
				if (!v.isDone())
					throw new IllegalArgumentException("validation no yet done!");
				else
				{
					List<Predictions> ps;
					if (v instanceof CV)
						ps = PredictionUtil.perFold(v.getResult());
					else
						ps = ListUtil.createList(v.getResult());

					for (Predictions p : ps)
					{
						int idx = rs.addResult();
						rs.setResultValue(idx, "ModelKey", v.getModel().getKey());
						rs.setResultValue(idx, "ModelName", v.getModel().getName());
						rs.setResultValue(idx, "Seed", v.getRandomSeed());
						if (v instanceof CV)
							rs.setResultValue(idx, "Fold", p.fold[0]);
						rs.setResultValue(idx, measure.toString(),
								PredictionUtil.getClassificationMeasure(p, measure,
										v.getDataSet().getPositiveClass()));
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
		String prefix = "";
		if (dataSet != null)
			prefix += dataSet.getKeyPrefix() + File.separator;
		prefix += getName() + "-repetitions" + repetitions + "-measure" + crit.getKeyPrefix();
		return prefix;
	}

	@Override
	public String getKeyContent()
	{
		return getKeyContent(repetitions, crit.getKeyContent(), dataSet, models, validation);
	}

	private List<Validation> validations;

	public List<Validation> getValidations()
	{
		if (validations == null)
		{
			validations = new ArrayList<>();
			for (int r = 0; r < repetitions; r++)
			{
				for (Model mod : models)
				{
					Validation v = (Validation) validation.cloneJob();
					v.setModel((Model) mod.cloneJob());
					v.setDataSet(dataSet);
					if (!v.isValid())
						continue;
					v.setRandomSeed((long) r);
					validations.add(v);
				}
			}
		}
		return validations;
	}

	@Override
	public Runnable nextJob() throws Exception
	{
		if (dataSet == null)
			throw new NullPointerException("set dataset first");

		List<Validation> vs = getValidations();
		if (vs.size() == 0)
			throw new IllegalStateException("nothing todo, probably all validations invalid");

		boolean allDone = true;
		for (final Validation v : vs)
		{
			if (!v.isDone())
			{
				allDone = false;
				final Runnable run = v.nextJob();
				if (run != null)
					return Printer.wrapRunnable(getName() + ": "
							+ ((getValidations().indexOf(v)) + 1) + "/" + getValidations().size(),
							run);
			}
		}

		if (allDone)
			return blockedJob(getName() + ": storing results", new Runnable()
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
		String bestModelKey = crit.selectBestModel(getValidations());
		setResult(bestModelKey);
		Model m = getBestModel();
		Printer.println("best model:\n" + m.getName());
	}

	public List<Predictions> getPredictions()
	{
		List<Predictions> l = new ArrayList<Predictions>();
		for (Validation v : getValidations())
		{
			//			System.out.println(cv.getName() + " " + cv.getModel().getName());
			if (!v.isDone())
				throw new IllegalArgumentException("validation not yet done!");
			else
				l.add(v.getResult());
		}
		if (l.size() != repetitions * models.length)
			throw new IllegalArgumentException();
		return l;
	}

	public Model getBestModel()
	{
		String result = getResult();
		for (Validation v : getValidations())
			if (v.getModel().getKey().toString().equals(result))
				return v.getModel();
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

	public void setRepetitions(int repetitions)
	{
		this.repetitions = repetitions;
	}

	public void setEvalCriterion(EvalCriterion crit)
	{
		this.crit = crit;
	}

	public void setValidation(Validation validation)
	{
		this.validation = validation;
	}

	public static void main(String[] args) throws Exception
	{
		final ValidationEval eval = new ValidationEval();
		Instances inst = new Instances(
				new FileReader("/home/martin/data/weka/nominal/breast-w.arff"));
		inst.setClassIndex(inst.numAttributes() - 1);
		eval.setDataSet(new WekaInstancesDataSet(inst, 1));
		eval.setModels(new RandomForestModel(), new NaiveBayesModel());
		//		CV cv = new CV();
		//		cv.setNumFolds(10);
		Holdout ho = new Holdout();
		ho.setSplitRatio(0.7);
		eval.setValidation(ho);
		eval.setRepetitions(1);
		eval.runSequentially();
		System.out.println("done: " + eval.getBestModel());
	}

	public DataSet getDataset()
	{
		return dataSet;
	}

}
