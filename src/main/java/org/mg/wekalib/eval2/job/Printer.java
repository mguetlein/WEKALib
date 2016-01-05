package org.mg.wekalib.eval2.job;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintStream;
import java.text.SimpleDateFormat;
import java.util.Date;

import org.mg.wekalib.eval2.persistance.DB;

public class Printer
{
	public static boolean PRINT_TO_SYSTEM_OUT = false;

	private static int indent = 0;

	private static ThreadLocal<PrintStream> out = new ThreadLocal<>();

	private static PrintStream out()
	{
		if (PRINT_TO_SYSTEM_OUT)
			return System.out;
		else
		{
			if (out.get() == null)
			{
				try
				{
					System.err.println("output goes to " + "jobs/out/" + DB.getThreadID());
					out.set(new PrintStream(new File("jobs/out/" + DB.getThreadID())));
				}
				catch (FileNotFoundException e)
				{
					throw new RuntimeException(e);
				}
			}
			return out.get();
		}
	}

	public static void increaseIndent()
	{
		indent += 2;
	}

	public static void decreaseIndent()
	{
		indent -= 2;
	}

	static SimpleDateFormat format = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss ");

	private static void printIndent()
	{
		out().print(format.format(new Date()));
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

	public static void println(Throwable t)
	{
		printIndent();
		out().println("ERROR");
		out().println(t.getMessage());
		t.printStackTrace(out());
	}

	public static Runnable wrapRunnable(String msg, Runnable r)
	{
		return wrapRunnable(msg, r, null);
	}

	public static Runnable wrapRunnable(final String msg, final Runnable r, final Runnable r2)
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
					if (r2 != null)
						r2.run();
				}
				finally
				{
					decreaseIndent();
				}
			}
		};
	}
}
