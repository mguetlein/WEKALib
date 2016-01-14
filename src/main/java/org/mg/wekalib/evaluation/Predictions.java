package org.mg.wekalib.evaluation;

import java.io.Serializable;

public class Predictions implements Serializable
{
	public static final long serialVersionUID = 1L;

	public double actual[];
	public double predicted[];
	public double confidence[];
	public int fold[];
	public int origIndex[];
}