/*
 * Copyright 2015 Jennifer Ling
 */
package twitter_crawler;

import twitter4j.*;

import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.List;

// get Twitter4J jar from http://twitter4j.org/en/index.html and add jar 
import twitter4j.Twitter;
import twitter4j.TwitterException;
import twitter4j.TwitterFactory;
import twitter4j.conf.ConfigurationBuilder;

import java.io.*;

/*
 * @author Jennifer Ling
 */
public class SearchTweets {	
	/*
	 * Get keys with Twitter Developer Account
	 * (see German instructions here:
	 * http://docs.aws.amazon.com/de_de/gettingstarted/latest/emr/getting-started-emr-sentiment-create-twitter-account.html
	 * )
	 */
	private static final String CONSUMER_KEY_1 = "<enter your key>";
	private static final String CONSUMER_SECRET_KEY_1 = "<enter your key>";
	private static final String ACCESS_TOKEN_1 = "<enter your token>";
	private static final String SECRET_ACCESS_TOKEN_1 = "<enter your token>";
	
	private static final String CONSUMER_KEY_2 = "<enter your key>";
	private static final String CONSUMER_SECRET_KEY_2 = "<enter your key>";
	private static final String ACCESS_TOKEN_2 = "<enter your token>";
	private static final String SECRET_ACCESS_TOKEN_2 = "<enter your token>";
	
	private static final String CONSUMER_KEY_3 = "<enter your key>";
	private static final String CONSUMER_SECRET_KEY_3 = "<enter your key>";
	private static final String ACCESS_TOKEN_3 = "<enter your token>";
	private static final String SECRET_ACCESS_TOKEN_3 = "<enter your token>";
	//create more login data if more keys are necessary and available.

	// twitter instance.
	private Twitter twitter;
	private String term;
	private String language;
	
	// Date format for file name. Uses date as unique file name.
	private final SimpleDateFormat formatter = new SimpleDateFormat ("yyyyMMdd-HHmmss");
	
	// constructor.
	public SearchTweets(ConfigurationBuilder cb, String term, String language) {
		
		// Make Twitter instance.
    	TwitterFactory tf = new TwitterFactory(cb.build());
    	twitter = tf.getInstance();
    	
    	this.term = term;
    	this.language = language;
    	
    	 
	}
	
    /**
     * Enter search term and needed language;
     * creates access with ConfigurationBuilder 
     * and searches for terms.
     * Every ConfigurationBuilder needs new keys and access tokens.
     * Here, three different Twitter Accounts are used 
     * due to tweet number restriction of 2800 Tweets for each query.
     * One Account can access the Twitter API every 15 minutes.
     * 
     * @param args search query
     * @throws InterruptedException 
     * @throws IOException 
     */
    public static void main(String[] args) throws InterruptedException, IOException { //  	
    	
    	/*
    	 *  every ConfigBuilder can be called multiple times with
    	 *  a time distance of 15 min.
    	 */
    	long start = System.currentTimeMillis();
    	run(createCB1(),"sarcasm", "en");
    	
    	run(createCB2(), "irony", "en");
    	
    	run(createCB3(), "ironic" ,"en");
    	
    	    	
    	long stop = System.currentTimeMillis();
    	long neededTime = stop - start;
    	
    	// thread sleeps since Twitter API only allows new query after 15 minutes.
    	if (neededTime < 900000){
    		long dif = 900000 - neededTime;
    		Thread.sleep(dif);
    	}
    	    	
    	start = System.currentTimeMillis();
    	
    	run(createCB1(),"sarcastic", "en");
    	
    	run(createCB2(), "humour", "en");
    	
    	run(createCB3(), "education" ,"en");
    	 	
    }

	private static void run(ConfigurationBuilder cb, String search_term, String language) throws FileNotFoundException,
			InterruptedException, IOException {
		SearchTweets st = new SearchTweets(cb, search_term, language); 	
    	st.searchTweets();
	}
	private static ConfigurationBuilder createCB1(){
		return createConfigurationBuilder(CONSUMER_KEY_1,CONSUMER_SECRET_KEY_1,ACCESS_TOKEN_1,SECRET_ACCESS_TOKEN_1);
	
	}
	
	private static ConfigurationBuilder createCB2(){
		return createConfigurationBuilder(CONSUMER_KEY_2,CONSUMER_SECRET_KEY_2,ACCESS_TOKEN_2,SECRET_ACCESS_TOKEN_2);
	}
	
	private static ConfigurationBuilder createCB3(){
		return createConfigurationBuilder(CONSUMER_KEY_3,CONSUMER_SECRET_KEY_3,ACCESS_TOKEN_3,SECRET_ACCESS_TOKEN_3);
	}
    
	private static ConfigurationBuilder createConfigurationBuilder(String consumerKey, String consumerSecret, String accessToken, String accessTokenSecret) {
		// Get access to Twitter.
    	ConfigurationBuilder cb = new ConfigurationBuilder();
    	
    	cb.setDebugEnabled(true)
    	  .setOAuthConsumerKey(consumerKey)
    	  .setOAuthConsumerSecret(consumerSecret)
    	  .setOAuthAccessToken(accessToken)
    	  .setOAuthAccessTokenSecret(accessTokenSecret);

		return cb;
	}
    
    public void searchTweets() throws FileNotFoundException, InterruptedException, IOException{
    	
    	// open output files (tab separated)
    	Date currentTime = new Date();
 	    OutputStream fos = new FileOutputStream("hashtag_" + term + "_" + formatter.format(currentTime) + ".csv");
 	    OutputStream bos = new BufferedOutputStream(fos);
 	    OutputStreamWriter osw = new OutputStreamWriter(bos, "UTF8");
 	        
	    String sep = "\t";
	    //print header in file
	    //osw.write("created at" + sep + "name" + sep + "username" + sep + "tweet_id" + sep + "processed_text" + sep + "geoLocation" + sep + "place \n");
        
 
	    // query + data for output.
        try {
        	        	
        	int tweet_number = 0;
        	
        	String query_term = "#" + term + " lang:" + language;
            Query query = new Query("#" + term + " lang:" + language); // "max_id:622886044971745280"
            
            QueryResult result;
            do {
                result = twitter.search(query);
                Thread.sleep(2000);
                List<Status> tweets = result.getTweets();
                Thread.sleep(2000);
                for (Status tweet : tweets) {           	   
                	tweet_number += 1;                	
                	
                	// preprocess text to remove paragraphs in the tweet because without them it's easier to process data.
                	String text = tweet.getText();
                	String processedText = text.replace("\n", " ");
                	processedText = processedText.replace("\r", " ");
                	// remove tabs in tweet because corpus will be saved tab-separated.
                	processedText = processedText.replace("\t", " ");
                	String user_name = tweet.getUser().getName().replace("\t", " ");
                	String screen_name = tweet.getUser().getScreenName().replace("\t", " ");
                	
                	
                	// if-else because place full name throws NullPointerException if there is no given place.
                	if (tweet.getPlace() != null){
                		
                		
                		String place = tweet.getPlace().getFullName();
                		
                		// Change readability of place information - format: city, country -> city in country
                		//String place = place.replace(",", " in");
                		
                		// write output file.
                		osw.write(tweet.getCreatedAt() + sep + user_name + sep + screen_name + sep + tweet.getId() + sep + processedText + sep + tweet.getGeoLocation() + sep + place + "\n");
                	}
                	// no given place information. 
                	else {
                		// write output file.
                		osw.write(tweet.getCreatedAt() + sep + user_name + sep + screen_name + sep + tweet.getId() + sep + processedText + sep + tweet.getGeoLocation() + sep + tweet.getPlace() + "\n");
                    	
                	}
                	
                	// Shows how much tweets are already found.
                	if (tweet_number % 100 == 0){                		
                		System.out.println("Saved " + tweet_number + " Tweets of category: " + term);
                	}
                }
            } while ((query = result.nextQuery()) != null);
            
            // gives number of tweets which where written in the files.
            System.out.println("All in all " + tweet_number + " Tweets are saved!" );
                       
            
            // close output files.
            osw.close();
         
        } catch (TwitterException te) {
            //te.printStackTrace();
            System.out.println("Failed to search tweets: " + te.getMessage());
            osw.close();

        } catch (FileNotFoundException fe) {
        	Exception e;
        }
    }
}

