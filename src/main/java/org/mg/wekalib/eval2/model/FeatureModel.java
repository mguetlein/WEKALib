package org.mg.wekalib.eval2.model;

import java.io.File;

import org.mg.javalib.io.KeyValueFileStore;
import org.mg.javalib.util.StringUtil;
import org.mg.javalib.util.ThreadUtil;
import org.mg.wekalib.eval2.data.DataSet;
import org.mg.wekalib.eval2.data.FeatureProvidedDataSet;
import org.mg.wekalib.eval2.job.DefaultJobOwner;
import org.mg.wekalib.eval2.job.FeatureProvider;
import org.mg.wekalib.eval2.job.Printer;
import org.mg.wekalib.evaluation.Predictions;

public class FeatureModel extends DefaultJobOwner<Predictions> implements Model
{
	FeatureProvider featureProvider;
	Model model;
	DataSet train;
	DataSet test;

	@Override
	public String getName()
	{
		return "FeatureModel (" + featureProvider.getName() + ", " + model.getName() + ")";
	}

	private FeatureProvider getFeatureModelFeatureProvider()
	{
		FeatureProvider feat = (FeatureProvider) featureProvider.cloneJob();
		feat.setTrainingDataset(train);
		feat.setTestDataset(test);
		return feat;
	}

	private Model getFeatureModelModel()
	{
		Model mod = (Model) model.cloneJob();
		FeatureProvider feat = getFeatureModelFeatureProvider();
		mod.setTrainingDataset(new FeatureProvidedDataSet(feat, true));
		mod.setTestDataset(new FeatureProvidedDataSet(feat, false));
		return mod;
	}

	@Override
	public String getKeyPrefix()
	{
		if (train != null && test != null)
			return getFeatureModelModel().getKeyPrefix();
		else
			return "FeatureModel" + File.separator + featureProvider.getKeyPrefix() + File.separator
					+ model.getKeyPrefix();
	}

	@Override
	public String getKeyContent()
	{
		if (train != null && test != null)
			return getFeatureModelModel().getKeyContent();
		else
			return getKeyContent(featureProvider, model);
	}

	@Override
	public Runnable nextJob() throws Exception
	{
		FeatureProvider feat = getFeatureModelFeatureProvider();
		if (!feat.isDone())
			return Printer.wrapRunnable("FeatureModel: compute features " + getName(),
					feat.nextJob());
		else
		{
			return Printer.wrapRunnable("FeatureModel: build model " + getName(),
					limitRuntime(getFeatureModelModel().nextJob()));
		}
	}

	@Override
	public Model cloneJob()
	{
		FeatureModel fm = new FeatureModel();
		fm.featureProvider = (FeatureProvider) featureProvider.cloneJob();
		fm.model = (Model) model.cloneJob();
		fm.train = train;
		fm.test = test;
		return fm;
	}

	public void setModel(Model model)
	{
		this.model = model;
	}

	public Model getModel()
	{
		return model;
	}

	public void setFeatureProvider(FeatureProvider featureProvider)
	{
		this.featureProvider = featureProvider;
	}

	public FeatureProvider getFeatureProvider()
	{
		return featureProvider;
	}

	@Override
	public void setTrainingDataset(DataSet train)
	{
		this.train = train;
	}

	@Override
	public DataSet getTrainingDataset()
	{
		return train;
	}

	@Override
	public void setTestDataset(DataSet test)
	{
		this.test = test;
	}

	@Override
	public String getAlgorithmParamsNice()
	{
		return model.getAlgorithmParamsNice();
	}

	@Override
	public String getAlgorithmShortName()
	{
		return model.getAlgorithmShortName();
	}

	@Override
	public boolean isValid(DataSet dataSet)
	{
		if (!featureProvider.isValid(dataSet))
			return false;
		if (tooSlow.contains(getTooSlowKey(dataSet.getName())))
			return false;
		return true;
	}

	public Runnable limitRuntime(final Runnable r)
	{
		if (r == null)
			return null;
		Runnable res = new Runnable()
		{
			@Override
			public void run()
			{
				final StringBuffer done = new StringBuffer();
				Thread slowThread = new Thread(new Runnable()
				{
					@Override
					public void run()
					{
						long start = System.currentTimeMillis();
						while (!done.toString().equals("done"))
						{
							ThreadUtil.sleep(333);
							if (System.currentTimeMillis() - start > MAX_RUNTIME)
							{
								tooSlow.store(getTooSlowKey(train.getName()), true);
								System.err.println("this alg is too slow on this data! "
										+ train.getName() + " " + model.getName() + " "
										+ featureProvider.getName());
								// HACK
								// this is not nice if multiple threads run
								// better would be make the model build thread abortable
								System.exit(1);
							}
						}
					}
				});
				slowThread.start();
				try
				{
					r.run();
				}
				finally
				{
					done.append("done");
				}
			}
		};
		return res;
	}

	public static long MAX_RUNTIME = 30 * 60 * 1000;
	private static KeyValueFileStore<String, Boolean> tooSlow = new KeyValueFileStore<>(
			System.getProperty("user.home") + "/results/cfpminer/tooSlow", false, false, null,
			true);

	private String getTooSlowKey(String datasetName)
	{
		if (featureProvider.getTrainingDataset() != null)
			throw new IllegalStateException();
		if (model.getTrainingDataset() != null)
			throw new IllegalStateException();
		String k = datasetName + File.separator + featureProvider.getKeyPrefix() + File.separator
				+ model.getKeyPrefix() + File.separator
				+ StringUtil.getMD5(getKeyContent(featureProvider, model));
		//		System.err.println(k);
		return k;
	}

}
