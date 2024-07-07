//
//  SuperTuxKart - a fun racing game with go-kart
//  Copyright (C) 2016 SuperTuxKart-Team
//
//  This program is free software; you can redistribute it and/or
//  modify it under the terms of the GNU General Public License
//  as published by the Free Software Foundation; either version 3
//  of the License, or (at your option) any later version.
//
//  This program is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU General Public License for more details.
//
//  You should have received a copy of the GNU General Public License
//  along with this program; if not, write to the Free Software
//  Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.

#include <iostream>
#include <stack>
#include <queue>
#include <vector>
#include "karts/controller/soccer_ai.hpp"
#include <fstream>
#include <sstream>
#include <cmath>
#include <string>
#include "items/attachment.hpp"
#include "items/powerup.hpp"
#include "karts/abstract_kart.hpp"
#include "karts/controller/kart_control.hpp"
#include "karts/kart_properties.hpp"
#include "modes/soccer_world.hpp"
#include "tracks/arena_graph.hpp"
#include "tracks/track.hpp"
#include "karts/controller/ai_shared_data.hpp"
//#include <lib/include/tensorflow/c/c_api.h>
//#include "libtorch/include/torch/csrc/api/include/torch/torch.h"
#include "onnxruntime/include/onnxruntime_cxx_api.h"

#include <cstdio>

#ifdef AI_DEBUG
#include "irrlicht.h"
#endif

#ifdef BALL_AIM_DEBUG
#include "graphics/camera.hpp"

#endif

std::stack<DataCollection> dataStack;
std::queue<DataCollection> dataCSV;

#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <iostream>

std::vector<std::vector<double>> loadCoefficients(const std::string& filename) {
    std::vector<std::vector<double>> coefficients;
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return coefficients;
    }

    std::string line;
    while (std::getline(file, line)) {
        std::vector<double> row;
        std::stringstream ss(line);
        std::string value;
        while (std::getline(ss, value, ',')) {
            try {
                row.push_back(std::stod(value));
            } catch (const std::invalid_argument& e) {
                std::cerr << "Invalid value found in CSV: " << value << std::endl;
                continue;
            } catch (const std::out_of_range& e) {
                std::cerr << "Value out of range in CSV: " << value << std::endl;
                continue;
            }
        }
        if (!row.empty()) {
            coefficients.push_back(row);
        }
    }

    file.close();
    return coefficients;
}


int predictTarget(const std::vector<double>& features, const std::vector<std::vector<double>>& coefficients) {
    std::vector<double> predictions(coefficients.size(), 0.0);

    // Multiply features by coefficients
    for (size_t i = 0; i < coefficients.size(); ++i) {
        for (size_t j = 0; j < features.size(); ++j) {
            predictions[i] += features[j] * coefficients[i][j];
        }
    }

    // Find the index of the maximum prediction value
    int maxIndex = 0;
    for (size_t i = 1; i < predictions.size(); ++i) {
        if (predictions[i] > predictions[maxIndex]) {
            maxIndex = i;
        }
    }

    return maxIndex; // Returns the index of the target class (0, 1, 2, or 3)
}

SoccerAI::SoccerAI(AbstractKart *kart)
        : ArenaAI(kart) ///HEREEEEEEEEEEE
{

    reset();

    //Load Coefficients
    std::string coefficients_file = "C:/Users/luizr/Desktop/stk-code-1.4/stk-code/build/bin/Debug/coefficients.csv";
    m_coefficients = loadCoefficients(coefficients_file);

#ifdef AI_DEBUG
    video::SColor col_debug(128, 128, 0, 0);
    video::SColor col_debug_next(128, 0, 128, 128);
    m_debug_sphere = irr_driver->addSphere(1.0f, col_debug);
    m_debug_sphere->setVisible(true);
    m_debug_sphere_next = irr_driver->addSphere(1.0f, col_debug_next);
    m_debug_sphere_next->setVisible(true);
#endif

#ifdef BALL_AIM_DEBUG
    video::SColor red(128, 128, 0, 0);
    video::SColor blue(128, 0, 0, 128);
    m_red_sphere = irr_driver->addSphere(1.0f, red);
    m_red_sphere->setVisible(true);
    m_blue_sphere = irr_driver->addSphere(1.0f, blue);
    m_blue_sphere->setVisible(true);
#endif

    m_world = dynamic_cast<SoccerWorld*>(World::getWorld());
    m_track = Track::getCurrentTrack();
    m_cur_team = m_world->getKartTeam(m_kart->getWorldKartId());
    m_opp_team = (m_cur_team == KART_TEAM_BLUE ?
        KART_TEAM_RED : KART_TEAM_BLUE);
        //OPEN FILE, READ
        // A = {std::vector}
        //populate A with the model from python
        ////loss function function to train model - (f(x) - y(i))2 - features - y- MSE
        //one hot enocding
        // 2 bots per team for the ball target problem

    // Don't call our own setControllerName, since this will add a
    // billboard showing 'AIBaseController' to the kart.
    Controller::setControllerName("SoccerAI");

}   // SoccerAI

//-----------------------------------------------------------------------------
SoccerAI::~SoccerAI()
{


#ifdef AI_DEBUG
    irr_driver->removeNode(m_debug_sphere);
    irr_driver->removeNode(m_debug_sphere_next);
#endif

#ifdef BALL_AIM_DEBUG
    irr_driver->removeNode(m_red_sphere);
    irr_driver->removeNode(m_blue_sphere);
#endif

}   //  ~SoccerAI

//-----------------------------------------------------------------------------
/** Resets the AI when a race is restarted.
 */
void SoccerAI::reset()
{
    ArenaAI::reset();

    m_overtake_ball = false;
    m_force_brake = false;
    m_chasing_ball = false;

    m_front_transform.setOrigin(m_kart->getFrontXYZ());
    m_front_transform.setBasis(m_kart->getTrans().getBasis());

}   // reset


//Target
TargetType targetType;

std::string kartTargetToString(TargetType kartTarget) {
    switch (kartTarget) {
        case BALL_NODE: return "BALL";
        case ITEM_NODE: return "ITEM";
        case CLOSEST_KART_NODE: return "CLOSEST_KART";
        case ATTACK_KART_NODE: return "ATTACK_NODE";
        default: return "UNKNOWN";
    }
}

void writeDataToCSV(const std::string& filename, const std::queue<DataCollection>& dataQueue) {
    std::ofstream outputFile(filename);
    
    if (!outputFile.is_open()) {
        std::cerr << "Failed to open file for writing: " << filename << std::endl;
        return;
    }
    
    // Header
    outputFile << "Kart ID,Kart Steer,KartAccel, Kart Speed,Kart Position X,Kart Position Y, Velocity X, Velocity Y, Velocity Z,Distance to Ball X, Distance to Ball Y, Distance to Ball Z, Kart Position Z,Ball Position X,Ball Position Y,Ball Position Z, Approaching GOal, Target, Is Scorer?, Team Goal?" << std::endl;
    
    std::queue<DataCollection> dataQueueCopy = dataQueue;
    while (!dataQueueCopy.empty()) 
    {
        DataCollection data = dataQueueCopy.front();
        outputFile << data.kartID << "," << data.kartSteer << "," << data.kartAccel << "," << data.kartSpeed << ","
                   << data.kartPos.x() << "," << data.kartPos.y() << "," << data.kartPos.z() << ","
                   << data.kartVelocity.x() << "," << data.kartVelocity.y() << "," << data.kartVelocity.z() << ","
                   << data.distanceToBall.x() << "," << data.distanceToBall.y() << "," << data.distanceToBall.z() << ","
                   << data.ballPos.x() << "," << data.ballPos.y() << "," << data.ballPos.z() << ","
                   << (data.isApproachingGoal ? "true" : "false") << "," // 
                   << kartTargetToString(data.kartTarget) << ","
                   << (data.isScorer ? "true" : "false") << ","
                   << (data.teamGoal ? "true" : "false")
                   << std::endl;
        dataQueueCopy.pop(); 
    }
    
    outputFile.close();
}



//-----------------------------------------------------------------------------
/** Update \ref m_front_transform for ball aiming functions, also make AI stop
 *  after goal.
 *  \param dt Time step size.
 */


bool isScorerTemp = false;
bool isTeamGoalTemp = false;
int tempScore = 0;

void SoccerAI::update(int ticks)
{


    int count = 0;

    static float timeElapsed = 0.0f;
    const int TICKS_PER_SECOND = 240;  

    // Increment time elapsed by converting ticks to seconds
    timeElapsed += static_cast<float>(ticks) / TICKS_PER_SECOND;

    const float dataCollectionInterval = 1.0f; // Collect data every second


#ifdef BALL_AIM_DEBUG
    Vec3 red = m_world->getBallAimPosition(KART_TEAM_RED);
    Vec3 blue = m_world->getBallAimPosition(KART_TEAM_BLUE);
    m_red_sphere->setPosition(red.toIrrVector());
    m_blue_sphere->setPosition(blue.toIrrVector());
#endif
    m_force_brake = false;

    m_chasing_ball = false;
    m_front_transform.setOrigin(m_kart->getFrontXYZ());
    m_front_transform.setBasis(m_kart->getTrans().getBasis());

    KartTeam kartTeam = (m_kart->getWorldKartId() == 0) ? KART_TEAM_RED : KART_TEAM_BLUE;
 

    if (m_world->isGoalPhase())
    {
    
        while (!dataStack.empty()) 
        {

            DataCollection q = dataStack.top();

            int scorerID = m_world->getScorer();
           
            q.isScorer = false;
            q.teamGoal = false;

            if (count < 10)
            {
                //Check who was the Scorer
                if (q.kartID == scorerID) 
                {
                    q.isScorer = true; 
                    q.teamGoal = true;

                }

                else if (scorerID == 1) //Check if it was a team goal 
                {
                    q.teamGoal = true;
                }


                //Variables to add on features vector for the prediction model on findtarget()
                isScorerTemp = q.isScorer;
                isTeamGoalTemp = q.teamGoal;

                std::cout << "Score ID: " << scorerID <<std::endl;
                std::cout <<"Test Red Score"<<m_world->getScore(KART_TEAM_RED) <<std::endl;
                std::cout <<"Test Red Score"<<m_world->getScore(KART_TEAM_BLUE) <<std::endl;
                std::cout <<"Team Goal" <<q.teamGoal<<std::endl;

                dataCSV.push(q); // Data selected to the CSV file - 10s before a goal - first 20 elements
                count++;

            }
            
            /* Debugging
            std::cout << "Kart ID: " << q.kartID << std::endl;
            std::cout << "Kart Steer: " << q.kartSteer << std::endl;
            std::cout << "Kart Speed: " << q.kartSpeed << std::endl;
            std::cout << "On Nitro: " << (q.onNitro ? "true" : "false") << std::endl;
            std::cout << "Kart Position: " << q.kartPos.x() <<"," <<q.kartPos.y()<<"," <<q.kartPos.z()<< std::endl;
            //std::cout<<"Skidding: " << q.kartSkidding.x() <<"," << q.kartSkidding.y()<<"," <<q.kartSkidding.z()<<std::endl;
            std::cout<<"Velocity: " << q.kartVelocity.x() <<"," << q.kartVelocity.y()<<"," <<q.kartVelocity.z()<<std::endl;
            std::cout<<"Distance to Ball: "<< q.distanceToBall.x() <<"," << q.distanceToBall.y()<<"," <<q.distanceToBall.z()<<std::endl;
            std::cout << "Ball Position: " << q.ballPos.x() <<", " << q.ballPos.y()<<"," <<q.ballPos.z()<<std::endl;
            std::cout << "Size: " <<dataStack.size() <<std::endl;
            std::cout << "Score rating 5" << q.scoreRating <<std::endl;
            std::cout << "isScorer" <<q.isScorer <<std::endl<<std::endl;
            std::cout << "isApproaching" <<q.isApproachingGoal <<std::endl; */
        
            // Pop the front element from the temporary queue
            dataStack.pop();


        }

        writeDataToCSV("output.csv", dataCSV);
        

        resetAfterStop();
        m_controls->setBrake(false);
        m_controls->setAccel(0.0f);
        AIBaseController::update(ticks);
        return;

    }
    
    else 
    {
        // Check if it's time to collect data (on avg, collect one data collection values from one kart per second)
        if (timeElapsed >= dataCollectionInterval)
        {

            timeElapsed -= dataCollectionInterval;

            // Push the data manually into the stack for the current kart
            dataStack.push({
                m_kart->getWorldKartId(),
                m_kart->getSteerPercent(),
                m_controls->getAccel(),
                m_kart->getSpeed(),
                m_kart->getFrontXYZ(),
                m_kart->getVelocity(),
                m_world->getBallPosition() - m_kart->getFrontXYZ(),
                m_world->getBallPosition(), // Convert Vec3 to X,Y,Z
                targetType,
                m_world->ballApproachingGoal(kartTeam),
                
            });

            findTarget(); // to update target

        }

    }

    ArenaAI::update(ticks);
}   // update

//-----------------------------------------------------------------------------
/** Find the closest kart around this AI, it won't find the kart with same
 *  team, consider_difficulty and find_sta are not used here.
 *  \param consider_difficulty If take current difficulty into account.
 *  \param find_sta If find \ref SpareTireAI only.
 */
void SoccerAI::findClosestKart(bool consider_difficulty, bool find_sta)
{
    float distance = 99999.9f;
    const unsigned int n = m_world->getNumKarts();
    int closest_kart_num = 0;

    for (unsigned int i = 0; i < n; i++)
    {
        const AbstractKart* kart = m_world->getKart(i);
        if (kart->isEliminated()) continue;

        if (kart->getWorldKartId() == m_kart->getWorldKartId())
            continue; // Skip the same kart

         if(kart->getWorldKartId() == 3)
            continue;

        if (m_world->getKartTeam(kart
            ->getWorldKartId()) == m_world->getKartTeam(m_kart
            ->getWorldKartId()))
            continue; // Skip the kart with the same team

        Vec3 d = kart->getXYZ() - m_kart->getXYZ();
        if (d.length_2d() <= distance)
        {
            distance = d.length_2d();
            closest_kart_num = i;
        }
    }

    m_closest_kart = m_world->getKart(closest_kart_num);
    m_closest_kart_node = m_world->getSectorForKart(m_closest_kart);
    m_closest_kart_point = m_closest_kart->getXYZ();

}   // findClosestKart

//-----------------------------------------------------------------------------
/** Find a suitable target to follow, it will first call
 *  \ref SoccerWorld::getBallChaser to check if this AI should go chasing the
 *  ball and try to score, otherwise it will call \ref tryCollectItem if
 *  needed. After that it will call \ref SoccerWorld::getAttacker to see if
 *  this AI should attack the kart in opposite team which is chasing the ball,
 *  if not go for the closest kart found by \ref findClosestKart.
 */
void SoccerAI::findTarget()
{
    std::cout<< "Time: " << m_world->getTime()<<std::endl;
    
    std::vector<double> features = {
        static_cast<double>(m_kart->getSteerPercent()),
        static_cast<double>(m_controls->getAccel()),
        static_cast<double>(m_kart->getSpeed()),
        m_kart->getFrontXYZ().x(), m_kart->getFrontXYZ().y(), m_kart->getFrontXYZ().z(),
        m_kart->getVelocity().x(), m_kart->getVelocity().y(), m_kart->getVelocity().z(),
        (m_world->getBallPosition() - m_kart->getFrontXYZ()).x(),
        (m_world->getBallPosition() - m_kart->getFrontXYZ()).y(),
        (m_world->getBallPosition() - m_kart->getFrontXYZ()).z(),
        m_world->getBallPosition().x(), m_world->getBallPosition().y(), m_world->getBallPosition().z(),
        static_cast<double>(m_world->ballApproachingGoal(m_cur_team)),
        static_cast<double>(isScorerTemp),
        static_cast<double>(isTeamGoalTemp)
    };

    //Prediction
    int predicted_target = predictTarget(features, m_coefficients);

    // Set predicted target
    switch (predicted_target) {
        case 0: targetType = BALL_NODE; 
        break;
        case 1: targetType = ITEM_NODE; 
        break;
        case 2: targetType = CLOSEST_KART_NODE;
        break;
        case 3: targetType = ATTACK_KART_NODE; 
        break;
        default: targetType = OTHER_NODE; 
        break;
    }
    findClosestKart(true/*consider_difficulty*/, false/*find_sta*/);


    //TEAM with my AI 
    if (m_world->getKartTeam(m_kart->getWorldKartId()) == KART_TEAM_RED)
    {

        if (predicted_target == 0) //Ball Node
        {
            m_target_point = determineBallAimingPosition();
            m_target_node  = m_world->getBallNode();

            std::cout<<"BALLL "<<std::endl;
            std::cout <<"Test Red Score"<<m_world->getScore(KART_TEAM_RED) <<std::endl;
            std::cout <<"Test Red Score"<<m_world->getScore(KART_TEAM_BLUE) <<std::endl;
            std::cout<< "Time: " << m_world->getTime()<<std::endl;
        
            return;
        }

        // Always reset this flag,
        // in case the ball chaser lost the ball somehow
        m_overtake_ball = false;

        if (predicted_target == 1) //
        {
            tryCollectItem(&m_target_point , &m_target_node);

            std::cout<<"ITEM "<<std::endl;

        }

        else if (predicted_target == 2)
        {
            // This AI will attack the other team ball chaser
            int id = m_world->getBallChaser(m_opp_team);
            const AbstractKart* kart = m_world->getKart(id);
            m_target_point = kart->getXYZ();
            m_target_node  = m_world->getSectorForKart(kart);

            std::cout<<"ATTACK"<<std::endl;

        }
        else
        {
            m_target_point = m_closest_kart_point;     //Closest Kart
            m_target_node  = m_closest_kart_node;

            std::cout<<"CLOSEST_KART"<<std::endl;

        }

    }

    //SuperTuxKart's AI
    
    else
    {
        std::cout<<"Estou aqui 2: "<<m_kart->getWorldKartId()<<std::endl;
        // Check if this AI kart is the one who will chase the ball
        if (m_world->getBallChaser(m_cur_team) == (signed)m_kart->getWorldKartId()) //Ball Node
        {
            m_target_point = determineBallAimingPosition();
            m_target_node  = m_world->getBallNode();

            /*if (m_kart->getWorldKartId() == 0) //this was use to the collect data 
            {
                //targetType = BALL_NODE;
                std::cout<<"BALLL "<<std::endl;
            }*/
           
        
            return;
        }

        // Always reset this flag,
        // in case the ball chaser lost the ball somehow
        m_overtake_ball = false;

        if (m_kart->getPowerup()->getType() == PowerupManager::POWERUP_NOTHING && //Item Node
            m_kart->getAttachment()->getType() != Attachment::ATTACH_SWATTER)
        {
            tryCollectItem(&m_target_point , &m_target_node);

        
            /*if (m_kart->getWorldKartId() == 0) //this was use to the collect data 
            {
                //targetType = ITEM_NODE;
                std::cout<<"ITEM "<<std::endl;
            }*/

        }

        else if (m_world->getAttacker(m_cur_team) == (signed)m_kart //Attacking Kart Node
            ->getWorldKartId())
        {
            // This AI will attack the other team ball chaser
            int id = m_world->getBallChaser(m_opp_team);
            const AbstractKart* kart = m_world->getKart(id);
            m_target_point = kart->getXYZ();
            m_target_node  = m_world->getSectorForKart(kart);

        
            /*if (m_kart->getWorldKartId() == 0) //this was use to the collect data 
            {
                //(m_world->getKartTeam(m_kart->getWorldKartId()) == KART_TEAM_RED)
                //targetType = ATTACK_KART_NODE;
                std::cout<<"ATTACK"<<std::endl;
            }*/

        }
        else
        {
            m_target_point = m_closest_kart_point;     //Closest Kart
            m_target_node  = m_closest_kart_node;

        
            /*if (m_kart->getWorldKartId() == 0) //this was use to the collect data 
            {
                //targetType = CLOSEST_KART_NODE;
                std::cout<<"CLOSEST_KART" <<std::endl;
            }*/

        }

    }

 
    /*This is for the variable in update()
    m_target_node_update = m_target_node;
    m_target_point_update = m_target_point;*/

}   // findTarget
//-----------------------------------------------------------------------------
/** Determine the point for aiming when try to steer or overtake the ball.
 *  AI will overtake the ball if the aiming position calculated by world is
 *  non-reachable.
 *  \return The coordinates to aim at.
 */

Vec3 SoccerAI::determineBallAimingPosition()
{
#ifdef BALL_AIM_DEBUG
    // Choose your favourite team to watch
    if (m_world->getKartTeam(m_kart->getWorldKartId()) == KART_TEAM_BLUE)
    {
        Camera *cam = Camera::getActiveCamera();
        cam->setMode(Camera::CM_NORMAL);
        cam->setKart(m_kart);
    }
#endif

    const Vec3& ball_aim_pos = m_world->getBallAimPosition(m_opp_team);
    const Vec3& orig_pos = m_world->getBallPosition();

    Vec3 ball_lc = m_front_transform.inverse()(orig_pos);
    Vec3 aim_lc = m_front_transform.inverse()(ball_aim_pos);

    // Too far from the ball,
    // use path finding from arena ai to get close
    // ie no extra braking is needed
    if (aim_lc.length_2d() > 10.0f) return ball_aim_pos;

    if (m_overtake_ball)
    {
        Vec3 overtake_lc;
        const bool can_overtake = determineOvertakePosition(ball_lc, aim_lc,
            &overtake_lc);
        if (!can_overtake)
        {
            m_overtake_ball = false;
            return ball_aim_pos;
        }
        else
            return m_front_transform(overtake_lc);
    }

    else
    {
        // Check whether the aim point is non-reachable
        // ie the ball is in front of the kart, which the aim position
        // is behind the ball , if so m_overtake_ball is true
        if (aim_lc.z() > 0 && aim_lc.z() > ball_lc.z())
        {
            if (isOvertakable(ball_lc))
            {
                m_overtake_ball = true;
                return ball_aim_pos;
            }
            else
            {
                // Stop a while to wait for overtaking, prevent own goal too
                // Only do that if the ball is moving
                if (!m_world->ballNotMoving())
                    m_force_brake = true;
                return ball_aim_pos;
            }
        }

        m_chasing_ball = true;
        // Check if reached aim point, which is behind aiming position and
        // in front of the ball, if so use another aiming method
        if (aim_lc.z() < 0 && ball_lc.z() > 0)
        {
            // Return the behind version of aim position, allow pushing to
            // ball towards the it
            return m_world->getBallAimPosition(m_opp_team, true/*reverse*/);
        }
    }

    // Otherwise keep steering until reach aim position
    return ball_aim_pos;

}   // determineBallAimingPosition

//-----------------------------------------------------------------------------
/** Used in \ref determineBallAimingPosition to test if AI can overtake the
 *  ball by testing distance.
 *  \param ball_lc Local coordinates of the ball.
 *  \return False if the kart is too close to the ball which can't overtake
 */
bool SoccerAI::isOvertakable(const Vec3& ball_lc)
{
    // No overtake if ball is behind
    if (ball_lc.z() < 0.0f) return false;

    // Circle equation: (x-a)2 + (y-b)2 = r2
    const float r2 = (ball_lc.length_2d() / 2) * (ball_lc.length_2d() / 2);
    const float a = ball_lc.x();
    const float b = ball_lc.z();

    // Check first if the kart is lies inside the circle, if so no tangent
    // can be drawn ( so can't overtake), minus 0.1 as epslion
    const float test_radius_2 = ((a * a) + (b * b)) - 0.1f;
    if (test_radius_2 < r2)
    {
        return false;
    }
    return true;

}   // isOvertakable

//-----------------------------------------------------------------------------
/** Used in \ref determineBallAimingPosition to pick a correct point to
 *  overtake the ball
 *  \param ball_lc Local coordinates of the ball.
 *  \param aim_lc Local coordinates of the aiming position.
 *  \param[out] overtake_lc Local coordinates of the overtaking position.
 *  \return True if overtaking is possible.
 */
bool SoccerAI::determineOvertakePosition(const Vec3& ball_lc,
                                         const Vec3& aim_lc,
                                         Vec3* overtake_lc)
{
    // This done by drawing a circle using the center of ball local coordinates
    // and the distance / 2 from kart to ball center as radius (which allows
    // more offset for overtaking), then find tangent line from kart (0, 0, 0)
    // to the circle. The intercept point will be used as overtake position

    // Check if overtakable at current location
    if (!isOvertakable(ball_lc)) return false;

    // Otherwise calculate the tangent
    // As all are local coordinates, so center is 0,0 which is y = mx for the
    // tangent equation, and the m (slope) can be calculated by puting y = mx
    // into the general form of circle equation x2 + y2 + Dx + Ey + F = 0
    // This means:  x2 + m2x2 + Dx + Emx + F = 0
    //                 (1+m2)x2 + (D+Em)x +F = 0
    // As only one root for it, so discriminant b2 - 4ac = 0
    // So:              (D+Em)2 - 4(1+m2)(F) = 0
    //           D2 + 2DEm +E2m2 - 4F - 4m2F = 0
    //      (E2 - 4F)m2 + (2DE)m + (D2 - 4F) = 0
    // Now solve the above quadratic equation using
    // x = -b (+/-) sqrt(b2 - 4ac) / 2a

    // Circle equation: (x-a)2 + (y-b)2 = r2
    const float r = ball_lc.length_2d() / 2;
    const float r2 = r * r;
    const float a = ball_lc.x();
    const float b = ball_lc.z();

    const float d = -2 * a;
    const float e = -2 * b;
    const float f = (d * d / 4) + (e * e / 4) - r2;
    const float discriminant = (2 * 2 * d * d * e * e) -
        (4 * ((e * e) - (4 * f)) * ((d * d) - (4 * f)));

    assert(discriminant > 0.0f);
    float t_slope_1 = (-(2 * d * e) + sqrtf(discriminant)) /
        (2 * ((e * e) - (4 * f)));
    float t_slope_2 = (-(2 * d * e) - sqrtf(discriminant)) /
        (2 * ((e * e) - (4 * f)));

    assert(!std::isnan(t_slope_1));
    assert(!std::isnan(t_slope_2));

    // Make the slopes in correct order, allow easier rotate later
    float slope_1 = 0.0f;
    float slope_2 = 0.0f;
    if ((t_slope_1 > 0 && t_slope_2 > 0) || (t_slope_1 < 0 && t_slope_2 < 0))
    {
        if (t_slope_1 > t_slope_2)
        {
            slope_1 = t_slope_1;
            slope_2 = t_slope_2;
        }
        else
        {
            slope_1 = t_slope_2;
            slope_2 = t_slope_1;
        }
    }
    else
    {
        if (t_slope_1 > t_slope_2)
        {
            slope_1 = t_slope_2;
            slope_2 = t_slope_1;
        }
        else
        {
            slope_1 = t_slope_1;
            slope_2 = t_slope_2;
        }
    }

    // Calculate two intercept points, as we already put y=mx into circle
    // equation and know that only one root for each slope, so x can be
    // calculated easily with -b / 2a
    // From (1+m2)x2 + (D+Em)x +F = 0:
    const float x1 = -(d + (e * slope_1)) / (2 * (1 + (slope_1 * slope_1)));
    const float x2 = -(d + (e * slope_2)) / (2 * (1 + (slope_2 * slope_2)));
    const float y1 = slope_1 * x1;
    const float y2 = slope_2 * x2;

    const Vec3 point1(x1, 0, y1);
    const Vec3 point2(x2, 0, y2);

    const float d1 = (point1 - aim_lc).length_2d();
    const float d2 = (point2 - aim_lc).length_2d();

    // Use the tangent closest to the ball aiming position to aim
    const bool use_tangent_one = d1 < d2;

    // Adjust x and y if r < ball diameter,
    // which will likely push to ball forward
    // Notice: we cannot increase the radius before, as the kart location
    // will likely lie inside the enlarged circle
    if (r < m_world->getBallDiameter())
    {
        // Constuctor a equation using y = (rotateSlope(old_m)) x which is
        // a less steep or steeper line, and find out the new adjusted position
        // using the distance to the original point * 2 at new line

        // Determine if the circle is drawn around the side of kart
        // ie point1 or point2 z() < 0, if so reverse the below logic
        const float m = ((point1.z() < 0 || point2.z() < 0) ?
            (use_tangent_one ? rotateSlope(slope_1, false/*rotate_up*/) :
            rotateSlope(slope_2, true/*rotate_up*/)) :
            (use_tangent_one ? rotateSlope(slope_1, true/*rotate_up*/) :
            rotateSlope(slope_2, false/*rotate_up*/)));

        // Calculate new distance from kart to new adjusted position
        const float dist = (use_tangent_one ? point1 : point2).length_2d() * 2;
        const float dist2 = dist * dist;

        // x2 + y2 = dist2
        // so y = m * sqrtf (dist2 - y2)
        // y = sqrtf(m2 * dist2 / (1 + m2))
        const float y = sqrtf((m * m * dist2) / (1 + (m * m)));
        const float x = y / m;
        *overtake_lc = Vec3(x, 0, y);
    }
    else
    {
        // Use the calculated position depends on distance to aim position
        if (use_tangent_one)
            *overtake_lc = point1;
        else
            *overtake_lc = point2;
    }

    return true;
}   // determineOvertakePosition

//-----------------------------------------------------------------------------
/** Used in \ref determineOvertakePosition to adjust the overtake position
 *  which is calculated by slope of line if it's too close.
 *  \param old_slope Old slope calculated.
 *  \param rotate_up If adjust the slope upwards.
 *  \return A newly calculated slope.
 */
float SoccerAI::rotateSlope(float old_slope, bool rotate_up)
{
    const float theta = atanf(old_slope) + (old_slope < 0 ? M_PI : 0);
    float new_theta = theta + (rotate_up ? M_PI / 6 : -M_PI /6);
    if (new_theta > ((M_PI / 2) - 0.02f) && new_theta < ((M_PI / 2) + 0.02f))
    {
        // Avoid almost tan 90
        new_theta = (M_PI / 2) - 0.02f;
    }
    // Check if over-rotated
    if (new_theta > M_PI)
        new_theta = M_PI - 0.1f;
    else if (new_theta < 0)
        new_theta = 0.1f;

    return tanf(new_theta);
}   // rotateSlope

//-----------------------------------------------------------------------------
int SoccerAI::getCurrentNode() const
{
    return m_world->getSectorForKart(m_kart);
}   // getCurrentNode

//-----------------------------------------------------------------------------
bool SoccerAI::isWaiting() const
{
    return m_world->isStartPhase();
}   // isWaiting

//-----------------------------------------------------------------------------
float SoccerAI::getKartDistance(const AbstractKart* kart) const
{
    return m_graph->getDistance(getCurrentNode(),
        m_world->getSectorForKart(kart));
}   // getKartDistance

//-----------------------------------------------------------------------------
bool SoccerAI::isKartOnRoad() const
{
    return m_world->isOnRoad(m_kart->getWorldKartId());
}   // isKartOnRoad
