package org.mg.wekalib.eval2.job;

import java.io.Serializable;

import org.mg.javalib.util.ArrayUtil;
import org.mg.javalib.util.ThreadUtil;

public abstract class DefaultJobOwner<R extends Serializable> implements JobOwner<R>
{
	private static Blocker BLOCKER = new Blocker();
	private static ResultProvider RESULTS = new ResultProvider();

	@Override
	public boolean isDone()
	{
		return RESULTS.contains(getKey());
	}

	@SuppressWarnings("unchecked")
	@Override
	public R getResult()
	{
		return (R) RESULTS.get(getKey());
	}

	protected void setResult(R r)
	{
		RESULTS.set(getKey(), r);
	}

	public static String getKey(Class<?> clazz, Object[] elements)
	{
		StringBuffer b = new StringBuffer();
		b.append(clazz.getSimpleName());
		for (Object o : ArrayUtil.flatten(elements))
		{
			b.append('#');
			if (o == null)
				b.append("null");
			else if (o instanceof KeyProvider)
				b.append(((KeyProvider) o).getKey());
			else if (o instanceof Enum<?> || o instanceof String || o instanceof Integer || o instanceof Double
					|| o instanceof Long || o instanceof Boolean)
				b.append(o.toString());
			else
				throw new IllegalArgumentException("Not a key provider: " + o + " " + o.getClass());
		}
		return b.toString();
	}

	protected String getKey(Object... elements)
	{
		return getKey(this.getClass(), elements);
	}

	protected Runnable blockedJob(final String msg, final Runnable r)
	{
		final String key = getKey();
		if (!BLOCKER.block(key))
			return null;
		return new Runnable()
		{
			@Override
			public void run()
			{
				try
				{
					Printer.println(msg + " (" + key + ")");
					r.run();
				}
				finally
				{
					BLOCKER.unblock(key);
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
				r.run();
			else
			{
				ThreadUtil.sleep(1000);
				System.out.println("wait until done");
			}
		}
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
			ThreadUtil.sleep(1000);
			System.out.println("wait until done");
		}
		return getResult();
	}
}
