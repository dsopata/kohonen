package Projekt.Zagadnienie4;

import java.util.Arrays;

import org.encog.neural.som.SOM;
import org.neuroph.core.data.DataSetRow;
import org.neuroph.nnet.Hopfield;
import org.neuroph.nnet.Kohonen;
import org.neuroph.nnet.learning.HopfieldLearning;
import org.neuroph.nnet.learning.KohonenLearning;

import Projekt.Data.Data;

public class Zadanie4 {
	
	private Kohonen kohenen;
	private KohonenLearning kohenenLearning;
	
	private Hopfield hopfield;
	private HopfieldLearning hopfieldLearning;
	
	private SOM som;
	
	long start = 0;
	long stop = 0;
	double executionTime = 0.0;
	
	public Zadanie4(Data data){
		System.out.println("KOHENEN");
		System.out.println();
		initKohenen(data);
		validateKohenen(data);
		
		System.out.println("HOPFIELD");
		System.out.println();
		initHopfield(data);
		validateHopfield(data);
		
	}

	private void validateHopfield(Data data) {
		System.out.println();
		for (DataSetRow dataRow : data.getValidatingSet().getRows()) {

			hopfield.setInput(dataRow.getInput());
			hopfield.calculate();
			hopfield.calculate();

			double[] networkOutput = hopfield.getOutput();
			System.out.println("Input: ");
			data.printMatrix(dataRow.getInput());
			System.out.println("Output: ");
			data.printMatrix(networkOutput);
		}
		System.out.println("Execution time: " + executionTime + " ms");
		System.out.println();
	}

	private void initHopfield(Data data) {
		hopfield= new Hopfield(35);
		hopfieldLearning = new HopfieldLearning();
		
		//hopfield.setLearningRule(hopfieldLearning);
		
		start = System.currentTimeMillis();
		hopfield.learn(data.getTrainingSetForUnsupervised());
		stop = System.currentTimeMillis();

		executionTime = stop - start;
		
	}

	private void validateKohenen(Data data) {
		System.out.println();
		for (DataSetRow dataRow : data.getValidatingSet().getRows()) {

			kohenen.setInput(dataRow.getInput());
			kohenen.calculate();

			double[] networkOutput = kohenen.getOutput();
			System.out.println("Input: ");
			data.printMatrix(dataRow.getInput());
			System.out.println(" Output: " + Arrays.toString(networkOutput));
		}
		System.out.println("Execution time: " + executionTime + " ms");
		System.out.println();
		
	}

	private void initKohenen(Data data) {
		kohenen= new Kohonen(35,1);
		kohenenLearning = new KohonenLearning();
		kohenenLearning.setIterations(20, 20);
		
		
		kohenen.setLearningRule(kohenenLearning);
		
		start = System.currentTimeMillis();
		kohenen.learn(data.getTrainingSetForUnsupervised());
		stop = System.currentTimeMillis();

		executionTime = stop - start;
	}
}
