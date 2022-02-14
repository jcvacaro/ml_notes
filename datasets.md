* pushshift.io Reddit We use a variant of Reddit discussions
* Following Humeau et al. (2019), we use a previously existing Reddit dataset extracted and obtained by a third party and made available on pushshift.io (Baumgartner et al., 2020), 
* training to generate a comment conditioned on the full thread leading up to the comment, 
* spanning 1.5B training examples from Reddit obtained from PushShift through July 2019. 
* The subreddits cover a vast range of topics
* We apply heuristic rules to filter the dataset with the goal of providing a cleaner training signal. 


* the ConvAI2 dataset (Zhang et al., 2018) focuses on personality and engaging the other speaker, 
    * used at the NeurIPS 2018 competition of the same name, 
    * is based on PersonaChat (Zhang et al., 2018; Dinan et al., 2020). 
    * The training data of 140k utterances
    * involves paired crowdworkers having a conversation where they get to know each other, 
    * each is given a role to play based on sentences describing their persona, which were also separately crowdsourced 
    * both speakers can see their own persona description, but cannot see their partner?s persona
    * The task thus involves getting to know the other speaker and engaging them in friendly conversation, both asking and answering questions
    * Models trained on this task are thus conditioned on the persona and the dialogue history, which are concatenated. 

* Empathetic Dialogues (Rashkin et al., 2019) focuses on empathy, 
    * consists of 50k utterances of crowdworker conversations grounded in an emotional situation. 
    * In each dialogue, one speaker describes a personal situation and the other plays a ?listener? role, displaying empathy during the discussion.
    * Trained models are measured playing the part of the empathetic listener. 

* Wizard of Wikipedia (Dinan et al., 2019c) focuses on knowledge. 
    * involves discussing a given topic in depth, where the goal is to both engage the partner as well as display expert knowledge (Dinan et al., 2019c). 
    * consists of 194k utterances over 1250 topics
    * each conversation begins with a randomly chosen topic. 
    * A retrieval system over Wikipedia was used from which the  dialogues were grounded during the human-humancrowdsourced conversations. 
    * The topics were also crowdsourced and range from e-books to toga parties to showers.

* Blended Skill Talk (Smith et al., 2020) 
    * aims to blend the previous three tasks to combine the skills from them 
    * a dialogue dataset of 76k utterances  was collected with a guided and unguided humanspeaker, 
    * where the guided speaker could select utterances  suggested by bots trained on the three individual tasks, see Figure 3. 
    * In each blended dialogue:
        * the model is provided a two sentence persona to condition on following PersonaChat, 
        * and additionally during one third of the conversations a WoW topic name as well (see Figure 3). 
    * During evaluations:
        * we equip our models with randomly chosen personas 
        * and, one third of the time, topics from this set as well, 

