package org.mg.wekalib.eval2.persistance;

import java.lang.management.ManagementFactory;

public class DB
{
	private static ThreadLocal<String> threadIDs = new ThreadLocal<>();
	private static Blocker BLOCKER;
	private static ResultProvider RESULTS;

	public static String getThreadID()
	{
		if (threadIDs.get() == null)
		{
			threadIDs.set(ManagementFactory.getRuntimeMXBean().getName() + "-" + Thread.currentThread().getId());
			System.err.println("thread key : " + threadIDs.get());
		}
		return threadIDs.get();
	}

	public static void setBlocker(Blocker b)
	{
		BLOCKER = b;
	}

	public static void setResultProvider(ResultProvider r)
	{
		RESULTS = r;
	}

	public static Blocker getBlocker()
	{
		if (BLOCKER == null)
			BLOCKER = new BlockerImpl();
		return BLOCKER;
	}

	public static ResultProvider getResultProvider()
	{
		if (RESULTS == null)
			RESULTS = new ResultProviderImpl();
		return RESULTS;
	}
}
