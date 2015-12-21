package org.mg.wekalib.eval2.job;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintStream;

public class Printer
{
	private static int indent = 0;

	private static ThreadLocal<PrintStream> out = new ThreadLocal<>();

	private static PrintStream out()
	{
		//return System.out;
		if (out.get() == null)
		{
			try
			{
				out.set(new PrintStream(new File("jobs/out/" + Blocker.getThreadID())));
			}
			catch (FileNotFoundException e)
			{
				throw new RuntimeException(e);
			}
		}
		return out.get();
	}

	public static void increaseIndent()
	{
		indent += 2;
	}

	public static void decreaseIndent()
	{
		indent -= 2;
	}

	private static void printIndent()
	{
		for (int i = 0; i < indent; i++)
			out().print(' ');
	}

	public static void println()
	{
		printIndent();
		out().println();
	}

	public static void println(Object o)
	{
		printIndent();
		out().println(o);
	}

	public static Runnable wrapRunnable(final String msg, final Runnable r)
	{
		if (r == null)
			return null;
		return new Runnable()
		{
			@Override
			public void run()
			{
				println(msg);
				increaseIndent();
				try
				{
					r.run();
				}
				finally
				{
					decreaseIndent();
				}
			}
		};
	}
}
