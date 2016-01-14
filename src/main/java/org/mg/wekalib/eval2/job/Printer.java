package org.mg.wekalib.eval2.job;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintStream;
import java.text.SimpleDateFormat;
import java.util.Date;

import org.mg.wekalib.eval2.persistance.DB;

public class Printer
{
	private static int indent = 0;

	private static String outfileDir = null;

	private static String outfileSuffix = null;

	private static ThreadLocal<PrintStream> out = new ThreadLocal<>();

	public static void setOutfile(String dir, String suffix)
	{
		outfileDir = dir;
		outfileSuffix = suffix;
	}

	private static PrintStream out()
	{
		if (outfileDir == null)
			return System.out;
		else
		{
			if (out.get() == null)
			{
				try
				{
					String file = outfileDir + "/" + DB.getThreadID();
					if (outfileSuffix != null)
						file += "_" + outfileSuffix;
					System.err.println("output goes to " + file);
					out.set(new PrintStream(new File(file)));
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

	public static void println_copyToError(Object o)
	{
		println(o);
		System.err.println(o);
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
