import java.awt.image.BufferedImage;
import java.io.File;
import java.io.*;
import javax.imageio.ImageIO;

import nl.captcha.Captcha;
import nl.captcha.backgrounds.*;
import nl.captcha.gimpy.*;
import nl.captcha.noise.*;
//import nl.captcha.text.*;

public class SimCapPro {
	public static void main(String[] args){
		BufferedWriter writer;
		try{	
		writer=new BufferedWriter(new FileWriter("image.txt"));
		for(int i=1;i<=600000;i++){
			Captcha captcha=new Captcha.Builder(200, 50).addText().addBorder().addBackground(new GradiatedBackgroundProducer()).gimp(new FishEyeGimpyRenderer()).addNoise().build();
			//System.out.println(captcha);
			//writer.write(captcha.getAnswer());
			//writer.newLine();
			BufferedImage img=captcha.getImage();
			String num=captcha.toString();
			//System.out.println(captcha.getAnswer());
			ImageIO.write(img,"jpg",new File("//home//gpu//tensorflow//tensorflow//ai-cap//pro_cap//captcha2//"+captcha.getAnswer()+".jpg"));
			}
		//writer.close();
		}catch(Exception e){
			System.out.println(e.toString());
		}
	}
}
