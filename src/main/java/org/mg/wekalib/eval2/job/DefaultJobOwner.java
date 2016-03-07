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
		boolean d = DB.getResultProvider().contains(getKey());
		//		if (d)
		//			System.out.println("REM is done: " + getKey());
		return d;
	}

	private R result;

	@SuppressWarnings("unchecked")
	@Override
	public R getResult()
	{
		if (result == null)
			result = (R) DB.getResultProvider().get(getKey());
		return result;
	}

	protected void setResult(R r)
	{
		if (DB.getBlocker() != null
				&& !DB.getBlocker().isBlockedByThread(getKey(), DB.getThreadID()))
			throw new IllegalStateException("job not blocked by this thread");
		if (isDone())
			throw new IllegalStateException("job already done");
		DB.getResultProvider().set(getKey(), r);
		result = r;
	}

	protected Runnable blockedJob(final String msg, final Runnable r)
	{
		final String key = getKey();
		if (DB.getBlocker() != null && !DB.getBlocker().block(key, DB.getThreadID()))
			return null;
		// it can happen that a job getsFinished right before/while blocking
		if (isDone())
		{
			if (DB.getBlocker() != null)
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

						if (DB.getBlocker() != null)
						{
							// wait after done and before unblocking to avoid
							// simultaneous unblocking by this thread and blocking by other thread
							ThreadUtil.sleep(1000);
							if (!isDone())
								throw new IllegalStateException("job not done.");
						}
					}
					catch (Exception e)
					{
						Printer.println(e);
						throw e;
					}
					finally
					{
						if (DB.getBlocker() != null)
							DB.getBlocker().unblock(key);
					}
				}
			};
	}

	@Override
	public R runSequentially() throws Exception
	{
		return runSequentially(false);
	}

	public R runSequentially(boolean silent) throws Exception
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
				if (!silent)
					Printer.println("wait until done");
			}
		}
		if (!silent)
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
