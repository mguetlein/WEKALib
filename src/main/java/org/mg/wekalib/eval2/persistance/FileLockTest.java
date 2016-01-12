package org.mg.wekalib.eval2.persistance;

import java.io.File;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.channels.FileLock;

import org.apache.commons.io.FileUtils;
import org.mg.javalib.util.ThreadUtil;

public class FileLockTest
{

	public static void main(String[] args) throws Exception
	{
		RandomAccessFile file = null;
		FileLock fileLock = null;
		try
		{
			file = new RandomAccessFile("FileToBeLocked", "rw");
			fileLock = file.getChannel().tryLock();
			ThreadUtil.sleep(3000);
			read();
			if (fileLock != null)
			{
				System.out.println("File is locked");
				write(args[0]);
				read();
			}
			else
				System.out.println("Could not lock file");
		}
		finally
		{
			if (fileLock != null)
			{
				fileLock.release();
			}
			file.close();
		}
	}

	static void read()
	{
		try
		{
			System.out
					.println("content> " + FileUtils.readFileToString(new File("FileToBeLocked")));
		}
		catch (Exception e)
		{
			e.printStackTrace();
		}
	}

	static void write(String s)
	{
		try
		{
			FileUtils.write(new File("FileToBeLocked"), s);
		}
		catch (IOException e)
		{
			e.printStackTrace();
		}
	}
}
