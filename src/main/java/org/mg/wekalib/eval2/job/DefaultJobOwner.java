package org.mg.wekalib.eval2.job;

import java.io.Serializable;

import org.mg.javalib.util.ThreadUtil;
import org.mg.wekalib.eval2.persistance.DB;

public abstract class DefaultJobOwner<R extends Serializable> extends DefaultComposedKeyProvider
		implements JobOwner<R>
{
	@Override
	public boolean isDone()
	{
		return DB.getResultProvider().contains(getKey());
	}

	@SuppressWarnings("unchecked")
	@Override
	public R getResult()
	{
		return (R) DB.getResultProvider().get(getKey());
	}

	protected void setResult(R r)
	{
		if (!DB.getBlocker().isBlockedByThread(getKey(), DB.getThreadID()))
			throw new IllegalStateException("job not blocked by this thread");
		if (isDone())
			throw new IllegalStateException("job already done");
		DB.getResultProvider().set(getKey(), r);
	}

	protected Runnable blockedJob(final String msg, final Runnable r)
	{
		final String key = getKey();
		if (!DB.getBlocker().block(key, DB.getThreadID()))
			return null;
		// it can happen that a job getsFinished right before/while blocking
		if (isDone())
		{
			DB.getBlocker().unblock(key);
			return null;
		}
		else
			return new Runnable()
			{
				@Override
				public void run()
				{
					try
					{
						if (isDone())
							throw new IllegalStateException("job already done.");
						Printer.println(msg + " (" + key + ")");
						r.run();
						// wait after done and before unblocking to avoid
						// simultaneous unblocking by this thread and blocking by other thread
						ThreadUtil.sleep(1000);
						if (!isDone())
							throw new IllegalStateException("job not done.");
					}
					catch (Exception e)
					{
						Printer.println(e);
						throw e;
					}
					finally
					{
						DB.getBlocker().unblock(key);
					}
				}
			};
	}

	@Override
	public R runSequentially() throws Exception
	{
		while (!isDone())
		{
			Runnable r = nextJob();
			if (r != null)
			{
				r.run();
			}
			else
			{
				ThreadUtil.sleep(10000);
				Printer.println("wait until done");
			}
		}
		Printer.println("done! " + getKey());
		return getResult();
	}

	@Override
	public R runParrallel(int numThreads) throws Exception
	{
		for (int i = 0; i < numThreads; i++)
		{
			Thread th = new Thread(new Runnable()
			{
				@Override
				public void run()
				{
					try
					{
						DefaultJobOwner.this.runSequentially();
					}
					catch (Exception e)
					{
						throw new RuntimeException(e);
					}
				}
			});
			th.start();
		}
		while (!isDone())
		{
			ThreadUtil.sleep(10000);
			Printer.println("wait until done");
		}
		Printer.println("done! " + getKey());
		return getResult();
	}
}
