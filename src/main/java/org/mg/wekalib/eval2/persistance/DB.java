package org.mg.wekalib.eval2.persistance;

import java.lang.management.ManagementFactory;

public class DB
{
	private static ThreadLocal<String> threadIDs = new ThreadLocal<>();
	private static Blocker BLOCKER = null;
	private static ResultProvider RESULTS = new ResultProviderImpl("/tmp/jobs/store",
			"/tmp/jobs/tmp");

	public static void init(ResultProvider p, Blocker b)
	{
		RESULTS = p;
		BLOCKER = b;
	}

	public static String getThreadID()
	{
		if (threadIDs.get() == null)
		{
			threadIDs.set(ManagementFactory.getRuntimeMXBean().getName() + "-"
					+ Thread.currentThread().getId());
			System.err.println("thread key : " + threadIDs.get());
		}
		return threadIDs.get();
	}

	public static Blocker getBlocker()
	{
		return BLOCKER;
	}

	public static ResultProvider getResultProvider()
	{
		return RESULTS;
	}

}
